// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

#include <hip/hip_runtime.h>

#include <drm/amdgpu_drm.h>
#include <libdrm/amdgpu.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_uuid.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_xclbin.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kDefaultRenderNode = "/dev/dri/renderD128";
constexpr const char* kDefaultXclbin =
  "/home/sadara/xdna-driver/build/vtd_extract/strx/validate_df_bandwidth.xclbin";
constexpr const char* kDefaultElf =
  "/home/sadara/xdna-driver/build/vtd_extract/strx/df_bw.elf";
constexpr const char* kDefaultKernel = "DPU";
constexpr uint64_t kTimeoutMs = 60000;
constexpr unsigned long long kNoBadIndex = 0xffffffffffffffffull;
constexpr uint64_t kDfBwProfileBufferBytes = 1024ull * 1024ull * 1024ull;
// HIP 7.13 exposes this enum in /opt/rocm/include, while this host's hipcc
// frontend includes an older /usr header. The runtime accepts the stable value.
constexpr int kHipDeviceAttributeDmaBufSupportedCompat = 93;

uint64_t
round_up(uint64_t value, uint64_t align)
{
  return ((value + align - 1) / align) * align;
}

bool
looks_like_df_bw_payload(const std::string& xclbin, const std::string& elf)
{
  return xclbin.find("validate_df_bandwidth") != std::string::npos ||
         elf.find("df_bw.elf") != std::string::npos;
}

uint64_t
now_us()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(
           clock::now().time_since_epoch())
    .count();
}

std::string
json_escape(const std::string& input)
{
  std::ostringstream os;
  for (unsigned char c : input) {
    switch (c) {
    case '"': os << "\\\""; break;
    case '\\': os << "\\\\"; break;
    case '\b': os << "\\b"; break;
    case '\f': os << "\\f"; break;
    case '\n': os << "\\n"; break;
    case '\r': os << "\\r"; break;
    case '\t': os << "\\t"; break;
    default:
      if (c < 0x20) {
        os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
           << static_cast<int>(c) << std::dec << std::setfill(' ');
      } else {
        os << static_cast<char>(c);
      }
    }
  }
  return os.str();
}

std::string
errno_string(int err)
{
  if (err < 0)
    err = -err;
  return std::strerror(err);
}

std::string
hip_error_string(hipError_t err)
{
  std::ostringstream os;
  os << hipGetErrorName(err) << ": " << hipGetErrorString(err);
  return os.str();
}

struct Stage {
  std::string name;
  std::string status;
  std::string detail;
  uint64_t us = 0;
};

struct Report {
  std::string mode;
  uint64_t size_bytes = 0;
  uint64_t alloc_bytes = 0;
  uint32_t iters = 0;
  std::string xclbin;
  std::string elf;
  std::string kernel;
  std::string render_node;
  int xrt_device = 0;
  int hip_device = 0;
  std::string hip_device_name;
  int hip_dmabuf_supported = -1;
  std::string result = "fail";
  std::string result_detail;
  uint64_t total_us = 0;
  uint64_t npu_us = 0;
  uint64_t hip_us = 0;
  uint64_t cpu_spot_us = 0;
  uint64_t expanded_bytes = 0;
  double npu_bytes_per_s = 0.0;
  double hip_bytes_per_s = 0.0;
  double end_to_end_bytes_per_s = 0.0;
  bool inverse_visibility_attempted = false;
  std::string inverse_visibility_status = "not_run";
  std::string inverse_visibility_detail;
  std::vector<Stage> stages;

  void
  add_stage(std::string name, std::string status, std::string detail = {}, uint64_t us = 0)
  {
    stages.push_back({std::move(name), std::move(status), std::move(detail), us});
  }

  std::string
  to_json() const
  {
    std::ostringstream os;
    os << "{\n";
    os << "  \"mode\": \"" << json_escape(mode) << "\",\n";
    os << "  \"size_bytes\": " << size_bytes << ",\n";
    os << "  \"alloc_bytes\": " << alloc_bytes << ",\n";
    os << "  \"iters\": " << iters << ",\n";
    os << "  \"xclbin\": \"" << json_escape(xclbin) << "\",\n";
    os << "  \"elf\": \"" << json_escape(elf) << "\",\n";
    os << "  \"kernel\": \"" << json_escape(kernel) << "\",\n";
    os << "  \"render_node\": \"" << json_escape(render_node) << "\",\n";
    os << "  \"xrt_device\": " << xrt_device << ",\n";
    os << "  \"hip_device\": " << hip_device << ",\n";
    os << "  \"hip_device_name\": \"" << json_escape(hip_device_name) << "\",\n";
    os << "  \"hip_dmabuf_supported\": " << hip_dmabuf_supported << ",\n";
    os << "  \"result\": \"" << json_escape(result) << "\",\n";
    os << "  \"result_detail\": \"" << json_escape(result_detail) << "\",\n";
    os << "  \"timings_us\": {\n";
    os << "    \"total\": " << total_us << ",\n";
    os << "    \"npu\": " << npu_us << ",\n";
    os << "    \"hip\": " << hip_us << ",\n";
    os << "    \"cpu_spot\": " << cpu_spot_us << "\n";
    os << "  },\n";
    os << "  \"throughput\": {\n";
    os << "    \"expanded_bytes\": " << expanded_bytes << ",\n";
    os << "    \"compressed_source_bytes\": " << expanded_bytes << ",\n";
    os << "    \"expanded_output_bytes\": " << expanded_bytes << ",\n";
    os << "    \"npu_bytes_per_s\": " << std::fixed << std::setprecision(3)
       << npu_bytes_per_s << ",\n";
    os << "    \"compressed_source_bytes_per_s\": " << npu_bytes_per_s << ",\n";
    os << "    \"expanded_output_bytes_per_s\": " << npu_bytes_per_s << ",\n";
    os << "    \"hip_bytes_per_s\": " << hip_bytes_per_s << ",\n";
    os << "    \"end_to_end_bytes_per_s\": " << end_to_end_bytes_per_s << "\n";
    os << "  },\n";
    os << "  \"inverse_visibility\": {\n";
    os << "    \"attempted\": " << (inverse_visibility_attempted ? "true" : "false") << ",\n";
    os << "    \"status\": \"" << json_escape(inverse_visibility_status) << "\",\n";
    os << "    \"detail\": \"" << json_escape(inverse_visibility_detail) << "\"\n";
    os << "  },\n";
    os << "  \"stages\": [\n";
    for (size_t i = 0; i < stages.size(); ++i) {
      const auto& s = stages[i];
      os << "    {\"name\":\"" << json_escape(s.name)
         << "\",\"status\":\"" << json_escape(s.status)
         << "\",\"detail\":\"" << json_escape(s.detail)
         << "\",\"us\":" << s.us << "}";
      if (i + 1 != stages.size())
        os << ",";
      os << "\n";
    }
    os << "  ]\n";
    os << "}\n";
    return os.str();
  }
};

struct Options {
  std::string mode = "single";
  uint64_t size_bytes = 4096;
  uint32_t iters = 1;
  std::string xclbin = kDefaultXclbin;
  std::string elf = kDefaultElf;
  std::string json_path;
  std::string render_node = kDefaultRenderNode;
  std::string kernel = kDefaultKernel;
  int xrt_device = 0;
  int hip_device = 0;
  bool skip_hip_compare = false;
  bool skip_cpu_spot = false;
};

void
usage(const char* argv0)
{
  std::cerr
    << "Usage: " << argv0 << " [options]\n"
    << "  --mode single|pingpong\n"
    << "  --size 4K|64K|1M|16M|64M|<bytes>\n"
    << "  --iters N\n"
    << "  --xclbin PATH\n"
    << "  --elf PATH\n"
    << "  --json PATH\n"
    << "  --kernel NAME           default: DPU\n"
    << "  --render-node PATH      default: /dev/dri/renderD128\n"
    << "  --xrt-device INDEX      default: 0\n"
    << "  --hip-device INDEX      default: 0\n"
    << "  --skip-hip-compare     import HIP memory but skip the GPU read/compare kernel\n"
    << "  --skip-cpu-spot        skip CPU spot-check reads from the output BO\n";
}

uint64_t
parse_size(const std::string& text)
{
  if (text.empty())
    throw std::runtime_error("empty --size");

  size_t pos = 0;
  while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])))
    ++pos;
  if (pos == 0)
    throw std::runtime_error("invalid --size '" + text + "'");

  uint64_t value = std::stoull(text.substr(0, pos));
  std::string suffix = text.substr(pos);
  std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });

  if (suffix.empty() || suffix == "B")
    return value;
  if (suffix == "K" || suffix == "KB")
    return value * 1024ull;
  if (suffix == "M" || suffix == "MB")
    return value * 1024ull * 1024ull;
  if (suffix == "G" || suffix == "GB")
    return value * 1024ull * 1024ull * 1024ull;
  throw std::runtime_error("unknown --size suffix in '" + text + "'");
}

Options
parse_args(int argc, char** argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };

    if (key == "--mode") {
      opt.mode = need_value("--mode");
    } else if (key == "--size") {
      opt.size_bytes = parse_size(need_value("--size"));
    } else if (key == "--iters") {
      opt.iters = static_cast<uint32_t>(std::stoul(need_value("--iters")));
    } else if (key == "--xclbin") {
      opt.xclbin = need_value("--xclbin");
    } else if (key == "--elf") {
      opt.elf = need_value("--elf");
    } else if (key == "--json") {
      opt.json_path = need_value("--json");
    } else if (key == "--render-node") {
      opt.render_node = need_value("--render-node");
    } else if (key == "--kernel") {
      opt.kernel = need_value("--kernel");
    } else if (key == "--xrt-device") {
      opt.xrt_device = std::stoi(need_value("--xrt-device"));
    } else if (key == "--hip-device") {
      opt.hip_device = std::stoi(need_value("--hip-device"));
    } else if (key == "--skip-hip-compare") {
      opt.skip_hip_compare = true;
    } else if (key == "--skip-cpu-spot") {
      opt.skip_cpu_spot = true;
    } else if (key == "-h" || key == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option '" + key + "'");
    }
  }

  if (opt.mode != "single" && opt.mode != "pingpong")
    throw std::runtime_error("--mode must be single or pingpong");
  if (opt.size_bytes == 0)
    throw std::runtime_error("--size must be non-zero");
  if ((opt.size_bytes % sizeof(uint32_t)) != 0)
    throw std::runtime_error("--size must be a multiple of 4 bytes");
  if (opt.iters == 0)
    throw std::runtime_error("--iters must be non-zero");
  return opt;
}

class UniqueFd {
public:
  UniqueFd() = default;
  explicit UniqueFd(int fd)
    : fd_(fd)
  {}

  ~UniqueFd()
  {
    reset();
  }

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;

  UniqueFd(UniqueFd&& other) noexcept
    : fd_(other.release())
  {}

  UniqueFd& operator=(UniqueFd&& other) noexcept
  {
    if (this != &other)
      reset(other.release());
    return *this;
  }

  int
  get() const
  {
    return fd_;
  }

  bool
  valid() const
  {
    return fd_ >= 0;
  }

  int
  release()
  {
    int fd = fd_;
    fd_ = -1;
    return fd;
  }

  void
  reset(int fd = -1)
  {
    if (fd_ >= 0)
      ::close(fd_);
    fd_ = fd;
  }

private:
  int fd_ = -1;
};

UniqueFd
dup_fd(int fd)
{
  int out = ::dup(fd);
  if (out < 0)
    throw std::runtime_error("dup(fd) failed: " + errno_string(errno));
  return UniqueFd(out);
}

void
maybe_preload_xrt(Report& report)
{
  const char* libs[] = {
    "/opt/xilinx/xrt/lib/libxrt_core.so.2",
    "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2",
    "/opt/xilinx/xrt/lib/libvxdna.so.1",
    "/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2",
  };

  for (const char* lib : libs) {
    uint64_t t0 = now_us();
    void* h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    uint64_t t1 = now_us();
    if (!h) {
      const char* err = dlerror();
      report.add_stage(std::string("preload:") + lib, "warn", err ? err : "dlopen failed", t1 - t0);
      continue;
    }
    report.add_stage(std::string("preload:") + lib, "pass", {}, t1 - t0);
  }
}

class AmdgpuDevice {
public:
  explicit AmdgpuDevice(const std::string& render_node)
  {
    fd_ = UniqueFd(::open(render_node.c_str(), O_RDWR | O_CLOEXEC));
    if (!fd_.valid())
      throw std::runtime_error("open(" + render_node + ") failed: " + errno_string(errno));

    uint32_t major = 0;
    uint32_t minor = 0;
    int ret = amdgpu_device_initialize(fd_.get(), &major, &minor, &dev_);
    if (ret)
      throw std::runtime_error("amdgpu_device_initialize failed: " + errno_string(ret));
  }

  ~AmdgpuDevice()
  {
    if (dev_)
      amdgpu_device_deinitialize(dev_);
  }

  AmdgpuDevice(const AmdgpuDevice&) = delete;
  AmdgpuDevice& operator=(const AmdgpuDevice&) = delete;

  amdgpu_device_handle
  get() const
  {
    return dev_;
  }

private:
  UniqueFd fd_;
  amdgpu_device_handle dev_ = nullptr;
};

class AmdgpuBo {
public:
  AmdgpuBo(amdgpu_device_handle dev, uint64_t size)
    : size_(size)
  {
    amdgpu_bo_alloc_request req = {};
    req.alloc_size = size_;
    req.phys_alignment = 4096;
    req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;
    req.flags = 0;

    int ret = amdgpu_bo_alloc(dev, &req, &bo_);
    if (ret)
      throw std::runtime_error("amdgpu_bo_alloc(GTT) failed: " + errno_string(ret));
  }

  ~AmdgpuBo()
  {
    if (mapped_)
      amdgpu_bo_cpu_unmap(bo_);
    if (bo_)
      amdgpu_bo_free(bo_);
  }

  AmdgpuBo(const AmdgpuBo&) = delete;
  AmdgpuBo& operator=(const AmdgpuBo&) = delete;

  UniqueFd
  export_dmabuf() const
  {
    uint32_t shared_handle = 0;
    int ret = amdgpu_bo_export(bo_, amdgpu_bo_handle_type_dma_buf_fd, &shared_handle);
    if (ret)
      throw std::runtime_error("amdgpu_bo_export(dma_buf_fd) failed: " + errno_string(ret));
    return UniqueFd(static_cast<int>(shared_handle));
  }

  void*
  cpu_map()
  {
    if (mapped_)
      return mapped_;
    void* ptr = nullptr;
    int ret = amdgpu_bo_cpu_map(bo_, &ptr);
    if (ret)
      throw std::runtime_error("amdgpu_bo_cpu_map failed: " + errno_string(ret));
    mapped_ = ptr;
    return mapped_;
  }

  void
  cpu_unmap()
  {
    if (!mapped_)
      return;
    int ret = amdgpu_bo_cpu_unmap(bo_);
    mapped_ = nullptr;
    if (ret)
      throw std::runtime_error("amdgpu_bo_cpu_unmap failed: " + errno_string(ret));
  }

  uint64_t
  size() const
  {
    return size_;
  }

private:
  amdgpu_bo_handle bo_ = nullptr;
  void* mapped_ = nullptr;
  uint64_t size_ = 0;
};

struct CheckResult {
  unsigned long long words;
  unsigned long long mismatches;
  unsigned long long first_bad;
  uint32_t got;
  uint32_t want;
};

__host__ __device__ uint32_t
expected_word(uint64_t index, uint32_t iter)
{
  uint32_t x = static_cast<uint32_t>(index);
  x ^= 0x9e3779b9u;
  x *= 0x85ebca6bu;
  x ^= (iter + 1u) * 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

__global__ void
compare_kernel(const uint32_t* data, uint64_t words, uint32_t iter, CheckResult* out)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = blockDim.x * gridDim.x;
  for (uint64_t i = tid; i < words; i += stride) {
    uint32_t got = data[i];
    uint32_t want = expected_word(i, iter);
    if (got != want) {
      atomicAdd(&out->mismatches, 1ull);
      unsigned long long old = atomicCAS(&out->first_bad, kNoBadIndex, static_cast<unsigned long long>(i));
      if (old == kNoBadIndex) {
        out->got = got;
        out->want = want;
      }
    }
  }
}

class HipMapping {
public:
  HipMapping(UniqueFd& fd, uint64_t size)
    : size_(size)
  {
    hipExternalMemoryHandleDesc mem_desc = {};
    mem_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = fd.get();
    mem_desc.size = size_;

    hipError_t err = hipImportExternalMemory(&external_, &mem_desc);
    if (err != hipSuccess)
      throw std::runtime_error("hipImportExternalMemory failed: " + hip_error_string(err));

    // OpaqueFd follows the CUDA/HIP external-memory ownership convention:
    // the runtime consumes the fd on successful import.
    fd.release();

    hipExternalMemoryBufferDesc buf_desc = {};
    buf_desc.offset = 0;
    buf_desc.size = size_;
    err = hipExternalMemoryGetMappedBuffer(&ptr_, external_, &buf_desc);
    if (err != hipSuccess)
      throw std::runtime_error("hipExternalMemoryGetMappedBuffer failed: " + hip_error_string(err));
  }

  ~HipMapping()
  {
    if (dev_result_)
      (void)hipFree(dev_result_);
    if (ptr_)
      (void)hipFree(ptr_);
    if (external_)
      (void)hipDestroyExternalMemory(external_);
  }

  HipMapping(const HipMapping&) = delete;
  HipMapping& operator=(const HipMapping&) = delete;

  void*
  ptr() const
  {
    return ptr_;
  }

  CheckResult
  compare(uint64_t bytes, uint32_t iter)
  {
    CheckResult init = {};
    init.words = bytes / sizeof(uint32_t);
    init.mismatches = 0;
    init.first_bad = kNoBadIndex;

    if (!dev_result_) {
      hipError_t err = hipMalloc(&dev_result_, sizeof(CheckResult));
      if (err != hipSuccess)
        throw std::runtime_error("hipMalloc(CheckResult) failed: " + hip_error_string(err));
    }

    hipError_t err = hipMemcpy(dev_result_, &init, sizeof(CheckResult), hipMemcpyHostToDevice);
    if (err != hipSuccess)
      throw std::runtime_error("hipMemcpy(H2D CheckResult) failed: " + hip_error_string(err));

    uint64_t words = init.words;
    int block = 256;
    int grid = static_cast<int>(std::min<uint64_t>(round_up(words, block) / block, 8192));
    if (grid < 1)
      grid = 1;

    hipLaunchKernelGGL(compare_kernel,
                       dim3(grid),
                       dim3(block),
                       0,
                       0,
                       static_cast<const uint32_t*>(ptr_),
                       words,
                       iter,
                       dev_result_);
    err = hipGetLastError();
    if (err != hipSuccess)
      throw std::runtime_error("compare_kernel launch failed: " + hip_error_string(err));
    err = hipDeviceSynchronize();
    if (err != hipSuccess)
      throw std::runtime_error("compare_kernel sync failed: " + hip_error_string(err));

    CheckResult result = {};
    err = hipMemcpy(&result, dev_result_, sizeof(CheckResult), hipMemcpyDeviceToHost);
    if (err != hipSuccess)
      throw std::runtime_error("hipMemcpy(D2H CheckResult) failed: " + hip_error_string(err));
    return result;
  }

private:
  uint64_t size_ = 0;
  hipExternalMemory_t external_ = nullptr;
  void* ptr_ = nullptr;
  CheckResult* dev_result_ = nullptr;
};

void
fill_input(xrt::bo& input, uint64_t bytes, uint32_t iter)
{
  auto* ptr = input.map<uint32_t*>();
  uint64_t words = bytes / sizeof(uint32_t);
  for (uint64_t i = 0; i < words; ++i)
    ptr[i] = expected_word(i, iter);
  input.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
}

std::string
cpu_spot_check(AmdgpuBo& bo, uint64_t bytes, uint32_t iter)
{
  auto* ptr = static_cast<const uint32_t*>(bo.cpu_map());
  uint64_t words = bytes / sizeof(uint32_t);
  std::vector<uint64_t> probes;
  probes.push_back(0);
  if (words > 1)
    probes.push_back(1);
  if (words > 2)
    probes.push_back(words / 2);
  if (words > 3)
    probes.push_back(words - 1);
  for (uint64_t i = 0; i < 60 && words > 0; ++i)
    probes.push_back((i * 11400714819323198485ull) % words);

  for (uint64_t idx : probes) {
    if (idx >= words)
      continue;
    uint32_t got = ptr[idx];
    uint32_t want = expected_word(idx, iter);
    if (got != want) {
      std::ostringstream os;
      os << "mismatch at word " << idx << ": got 0x" << std::hex << got
         << " want 0x" << want;
      bo.cpu_unmap();
      return os.str();
    }
  }

  bo.cpu_unmap();
  return {};
}

struct XrtContext {
  explicit XrtContext(const Options& opt)
    : device(opt.xrt_device)
    , xclbin(opt.xclbin)
    , uuid(device.register_xclbin(xclbin))
    , elf(opt.elf)
    , module(elf)
    , hwctx(device, uuid)
    , kernel(xrt::ext::kernel(hwctx, module, opt.kernel))
  {}

  xrt::device device;
  xrt::xclbin xclbin;
  xrt::uuid uuid;
  xrt::elf elf;
  xrt::module module;
  xrt::hw_context hwctx;
  xrt::kernel kernel;
};

void
set_df_bw_args(xrt::run& run, xrt::bo& input, xrt::bo& output)
{
  run.set_arg(0, 3);
  run.set_arg(1, 0);
  run.set_arg(2, 0);
  run.set_arg(3, input);
  run.set_arg(5, output);
  run.set_arg(6, 0);
  run.set_arg(7, 0);
}

void
run_npu_once(XrtContext& ctx, xrt::bo& input, xrt::bo& output, uint64_t bytes)
{
  xrt::run run(ctx.kernel);
  set_df_bw_args(run, input, output);
  run.start();
  auto state = run.wait(kTimeoutMs);
  if (state != ERT_CMD_STATE_COMPLETED) {
    std::ostringstream os;
    os << "xrt run.wait returned state " << static_cast<int>(state);
    throw std::runtime_error(os.str());
  }
  output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
}

struct Slot {
  std::unique_ptr<AmdgpuBo> bo;
  UniqueFd export_fd;
  UniqueFd xrt_fd;
  UniqueFd hip_fd;
  std::unique_ptr<xrt::bo> xrt_bo;
  std::unique_ptr<HipMapping> hip;
  bool hip_ok = false;
  std::string hip_error;
};

void
try_inverse_visibility_diag(XrtContext& ctx, uint64_t bytes, Report& report)
{
  report.inverse_visibility_attempted = true;
  uint64_t alloc_bytes = round_up(bytes, 4096);
  uint64_t t0 = now_us();
  try {
    xrt::bo xrt_owned(ctx.device, alloc_bytes, xrt::bo::flags::host_only, 0);
    int exported = xrt_owned.export_buffer();
    UniqueFd hip_fd = dup_fd(exported);
    HipMapping hip(hip_fd, alloc_bytes);
    report.inverse_visibility_status = "visibility_only";
    report.inverse_visibility_detail = "XRT-owned BO exported to HIP imported successfully";
    report.add_stage("inverse_xrt_export_to_hip", "visibility_only", report.inverse_visibility_detail, now_us() - t0);
  } catch (const std::exception& e) {
    report.inverse_visibility_status = "fail";
    report.inverse_visibility_detail = e.what();
    report.add_stage("inverse_xrt_export_to_hip", "fail", e.what(), now_us() - t0);
  }
}

Slot
make_slot(AmdgpuDevice& gpu, XrtContext& ctx, uint64_t alloc_bytes, bool hip_dmabuf_supported, Report& report, int index)
{
  Slot slot;
  auto suffix = std::string("[") + std::to_string(index) + "]";

  uint64_t t0 = now_us();
  slot.bo = std::make_unique<AmdgpuBo>(gpu.get(), alloc_bytes);
  report.add_stage("amdgpu_alloc_gtt" + suffix, "pass", {}, now_us() - t0);

  t0 = now_us();
  slot.export_fd = slot.bo->export_dmabuf();
  report.add_stage("amdgpu_export_dmabuf" + suffix, "pass", {}, now_us() - t0);

  slot.xrt_fd = dup_fd(slot.export_fd.get());
  slot.hip_fd = dup_fd(slot.export_fd.get());

  t0 = now_us();
  try {
    slot.xrt_bo = std::make_unique<xrt::bo>(ctx.device, static_cast<xrt::bo::export_handle>(slot.xrt_fd.get()));
    report.add_stage("xrt_import_amdgpu_dmabuf" + suffix, "pass", {}, now_us() - t0);
  } catch (const std::exception& e) {
    report.add_stage("xrt_import_amdgpu_dmabuf" + suffix, "fail", e.what(), now_us() - t0);
    try_inverse_visibility_diag(ctx, alloc_bytes, report);
    throw;
  }

  if (!hip_dmabuf_supported) {
    slot.hip_ok = false;
    slot.hip_error = "hipDeviceAttributeDmaBufSupported is false";
    report.add_stage("hip_import_amdgpu_dmabuf" + suffix, "skip", slot.hip_error);
    return slot;
  }

  t0 = now_us();
  try {
    slot.hip = std::make_unique<HipMapping>(slot.hip_fd, alloc_bytes);
    slot.hip_ok = true;
    report.add_stage("hip_import_amdgpu_dmabuf" + suffix, "pass", {}, now_us() - t0);
  } catch (const std::exception& e) {
    slot.hip_ok = false;
    slot.hip_error = e.what();
    report.add_stage("hip_import_amdgpu_dmabuf" + suffix, "fail", slot.hip_error, now_us() - t0);
  }

  return slot;
}

bool
single_iteration(XrtContext& ctx,
                 xrt::bo& input,
                 Slot& slot,
                 uint64_t bytes,
                 uint32_t iter,
                 Report& report,
                 const std::string& label,
                 bool skip_hip_compare,
                 bool skip_cpu_spot)
{
  uint64_t t0 = now_us();
  fill_input(input, bytes, iter);
  report.add_stage("input_fill_sync" + label, "pass", {}, now_us() - t0);

  t0 = now_us();
  run_npu_once(ctx, input, *slot.xrt_bo, bytes);
  uint64_t npu_us = now_us() - t0;
  report.npu_us += npu_us;
  report.add_stage("npu_run_sync" + label, "pass", {}, npu_us);

  if (skip_cpu_spot) {
    report.add_stage("cpu_spot_check" + label, "skip", "--skip-cpu-spot");
  } else {
    t0 = now_us();
    std::string cpu_err = cpu_spot_check(*slot.bo, bytes, iter);
    uint64_t cpu_us = now_us() - t0;
    report.cpu_spot_us += cpu_us;
    if (!cpu_err.empty()) {
      report.add_stage("cpu_spot_check" + label, "fail", cpu_err, cpu_us);
      return false;
    }
    report.add_stage("cpu_spot_check" + label, "pass", {}, cpu_us);
  }

  if (!slot.hip_ok) {
    report.add_stage("hip_compare" + label, "skip", slot.hip_error);
    return true;
  }
  if (skip_hip_compare) {
    report.add_stage("hip_compare" + label, "skip", "--skip-hip-compare");
    return true;
  }

  t0 = now_us();
  CheckResult check = slot.hip->compare(bytes, iter);
  uint64_t hip_us = now_us() - t0;
  report.hip_us += hip_us;
  if (check.mismatches != 0) {
    std::ostringstream os;
    os << "mismatches=" << check.mismatches << " first_bad=" << check.first_bad
       << " got=0x" << std::hex << check.got << " want=0x" << check.want;
    report.add_stage("hip_compare" + label, "fail", os.str(), hip_us);
    return false;
  }
  report.add_stage("hip_compare" + label, "pass", {}, hip_us);
  return true;
}

bool
run_single(XrtContext& ctx,
           AmdgpuDevice& gpu,
           xrt::bo& input,
           uint64_t bytes,
           uint64_t alloc_bytes,
           bool hip_dmabuf_supported,
           Report& report,
           bool skip_hip_compare,
           bool skip_cpu_spot)
{
  Slot slot = make_slot(gpu, ctx, alloc_bytes, hip_dmabuf_supported, report, 0);
  bool ok = true;
  for (uint32_t iter = 0; iter < report.iters; ++iter)
    ok = single_iteration(ctx, input, slot, bytes, iter, report, "[0]", skip_hip_compare, skip_cpu_spot) && ok;

  if (!ok)
    return false;
  if (slot.hip_ok && !skip_hip_compare) {
    report.result = "pass";
    report.result_detail = "amdgpu GTT dma-buf imported by XRT and HIP; NPU write and HIP compare passed";
  } else {
    report.result = "partial";
    report.result_detail = skip_hip_compare
      ? "NPU write completed; HIP compare intentionally skipped"
      : "NPU write to amdgpu BO passed CPU spot-check; HIP stage did not pass: " + slot.hip_error;
  }
  return true;
}

bool
run_pingpong(XrtContext& ctx,
             AmdgpuDevice& gpu,
             xrt::bo& input,
             uint64_t bytes,
             uint64_t alloc_bytes,
             bool hip_dmabuf_supported,
             Report& report,
             bool skip_hip_compare,
             bool skip_cpu_spot)
{
  Slot slots[2] = {
    make_slot(gpu, ctx, alloc_bytes, hip_dmabuf_supported, report, 0),
    make_slot(gpu, ctx, alloc_bytes, hip_dmabuf_supported, report, 1),
  };

  // First prove both imported buffers sequentially before timing overlap.
  for (uint32_t i = 0; i < 2; ++i) {
    if (!single_iteration(ctx,
                          input,
                          slots[i],
                          bytes,
                          i,
                          report,
                          "[warmup" + std::to_string(i) + "]",
                          skip_hip_compare,
                          skip_cpu_spot))
      return false;
  }

  bool hip_available = slots[0].hip_ok && slots[1].hip_ok && !skip_hip_compare;
  bool ok = true;
  uint32_t completed_slot = 0;
  bool have_completed = false;

  for (uint32_t iter = 0; iter < report.iters; ++iter) {
    uint32_t cur = iter & 1u;
    uint32_t prev = cur ^ 1u;

    uint64_t t0 = now_us();
    fill_input(input, bytes, iter + 1000);
    report.add_stage("input_fill_sync[pp" + std::to_string(iter) + "]", "pass", {}, now_us() - t0);

    xrt::run run(ctx.kernel);
    set_df_bw_args(run, input, *slots[cur].xrt_bo);

    t0 = now_us();
    run.start();

    if (have_completed && hip_available) {
      uint64_t hip_t0 = now_us();
      CheckResult check = slots[prev].hip->compare(bytes, iter + 999);
      uint64_t hip_elapsed = now_us() - hip_t0;
      report.hip_us += hip_elapsed;
      if (check.mismatches != 0) {
        std::ostringstream os;
        os << "mismatches=" << check.mismatches << " first_bad=" << check.first_bad;
        report.add_stage("hip_compare_overlap[pp" + std::to_string(iter - 1) + "]", "fail", os.str(), hip_elapsed);
        ok = false;
      } else {
        report.add_stage("hip_compare_overlap[pp" + std::to_string(iter - 1) + "]", "pass", {}, hip_elapsed);
      }
    }

    auto state = run.wait(kTimeoutMs);
    uint64_t npu_elapsed = now_us() - t0;
    report.npu_us += npu_elapsed;
    if (state != ERT_CMD_STATE_COMPLETED) {
      std::ostringstream os;
      os << "xrt run.wait returned state " << static_cast<int>(state);
      report.add_stage("npu_run[pp" + std::to_string(iter) + "]", "fail", os.str(), npu_elapsed);
      return false;
    }
    slots[cur].xrt_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
    report.add_stage("npu_run_sync[pp" + std::to_string(iter) + "]", "pass", {}, npu_elapsed);

    completed_slot = cur;
    have_completed = true;
  }

  if (have_completed && hip_available) {
    uint64_t t0 = now_us();
    CheckResult check = slots[completed_slot].hip->compare(bytes, report.iters + 999);
    uint64_t hip_elapsed = now_us() - t0;
    report.hip_us += hip_elapsed;
    if (check.mismatches != 0) {
      std::ostringstream os;
      os << "mismatches=" << check.mismatches << " first_bad=" << check.first_bad;
      report.add_stage("hip_compare_overlap[last]", "fail", os.str(), hip_elapsed);
      ok = false;
    } else {
      report.add_stage("hip_compare_overlap[last]", "pass", {}, hip_elapsed);
    }
  }

  if (ok && hip_available) {
    report.result = "pass";
    report.result_detail = "ping-pong NPU fill and HIP compare completed with CPU-managed overlap";
  } else if (ok) {
    report.result = "partial";
    report.result_detail = skip_hip_compare
      ? "ping-pong NPU path completed; HIP compare intentionally skipped"
      : "ping-pong NPU path completed, but HIP import was unavailable";
  }
  return ok;
}

void
write_report(const Report& report, const std::string& path)
{
  if (path.empty())
    return;
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("failed to open JSON report path: " + path);
  out << report.to_json();
}

} // namespace

int
main(int argc, char** argv)
{
  Report report;
  int exit_code = 1;

  try {
    Options opt = parse_args(argc, argv);
    report.mode = opt.mode;
    report.size_bytes = opt.size_bytes;
    report.alloc_bytes = round_up(opt.size_bytes, 4096);
    if (looks_like_df_bw_payload(opt.xclbin, opt.elf))
      report.alloc_bytes = std::max<uint64_t>(report.alloc_bytes, kDfBwProfileBufferBytes);
    report.iters = opt.iters;
    report.xclbin = opt.xclbin;
    report.elf = opt.elf;
    report.kernel = opt.kernel;
    report.render_node = opt.render_node;
    report.xrt_device = opt.xrt_device;
    report.hip_device = opt.hip_device;
    if (opt.skip_hip_compare)
      report.add_stage("option", "info", "--skip-hip-compare");
    if (opt.skip_cpu_spot)
      report.add_stage("option", "info", "--skip-cpu-spot");

    uint64_t total_t0 = now_us();
    maybe_preload_xrt(report);

    uint64_t t0 = now_us();
    hipError_t hip_err = hipSetDevice(opt.hip_device);
    if (hip_err != hipSuccess)
      throw std::runtime_error("hipSetDevice failed: " + hip_error_string(hip_err));
    hipDeviceProp_t props = {};
    hip_err = hipGetDeviceProperties(&props, opt.hip_device);
    if (hip_err != hipSuccess)
      throw std::runtime_error("hipGetDeviceProperties failed: " + hip_error_string(hip_err));
    report.hip_device_name = props.name;
    int dmabuf_supported = 1;
    std::string hip_attr_detail;
    hip_err = hipDeviceGetAttribute(&dmabuf_supported,
                                    static_cast<hipDeviceAttribute_t>(kHipDeviceAttributeDmaBufSupportedCompat),
                                    opt.hip_device);
    if (hip_err == hipSuccess) {
      report.hip_dmabuf_supported = dmabuf_supported;
    } else {
      report.hip_dmabuf_supported = -1;
      hip_attr_detail = " (DmaBufSupported attribute query failed: " + hip_error_string(hip_err) +
                        "; import will be attempted)";
    }
    std::string hip_detail = props.name;
    if (report.hip_dmabuf_supported < 0)
      hip_detail += hip_attr_detail;
    report.add_stage("hip_device_preflight", "pass", hip_detail, now_us() - t0);

    t0 = now_us();
    AmdgpuDevice gpu(opt.render_node);
    report.add_stage("amdgpu_open", "pass", opt.render_node, now_us() - t0);

    t0 = now_us();
    XrtContext xrt(opt);
    report.add_stage("xrt_context", "pass", {}, now_us() - t0);

    t0 = now_us();
    xrt::bo input(xrt.device, report.alloc_bytes, xrt::bo::flags::host_only, 0);
    report.add_stage("xrt_input_host_bo", "pass", {}, now_us() - t0);

    bool ok = false;
    if (opt.mode == "single") {
      ok = run_single(xrt,
                      gpu,
                      input,
                      report.size_bytes,
                      report.alloc_bytes,
                      dmabuf_supported != 0,
                      report,
                      opt.skip_hip_compare,
                      opt.skip_cpu_spot);
    } else {
      ok = run_pingpong(xrt,
                        gpu,
                        input,
                        report.size_bytes,
                        report.alloc_bytes,
                        dmabuf_supported != 0,
                        report,
                        opt.skip_hip_compare,
                        opt.skip_cpu_spot);
    }

    report.total_us = now_us() - total_t0;
    report.expanded_bytes = report.size_bytes * report.iters;
    if (report.npu_us > 0)
      report.npu_bytes_per_s = static_cast<double>(report.expanded_bytes) * 1000000.0 / static_cast<double>(report.npu_us);
    if (report.hip_us > 0)
      report.hip_bytes_per_s = static_cast<double>(report.expanded_bytes) * 1000000.0 / static_cast<double>(report.hip_us);
    if (report.total_us > 0)
      report.end_to_end_bytes_per_s = static_cast<double>(report.expanded_bytes) * 1000000.0 / static_cast<double>(report.total_us);

    if (!ok && report.result == "fail")
      report.result_detail = "one or more required stages failed";
    bool intentional_partial = opt.skip_hip_compare;
    exit_code = ok && (report.result == "pass" || intentional_partial) ? 0 : 2;
    write_report(report, opt.json_path);

    std::cout << report.result << ": " << report.result_detail << "\n";
    std::cout << "json: " << (opt.json_path.empty() ? "(not requested)" : opt.json_path) << "\n";
    return exit_code;
  } catch (const std::exception& e) {
    report.result = "fail";
    report.result_detail = e.what();
    report.add_stage("fatal", "fail", e.what());
    try {
      write_report(report, [&]() -> std::string {
        for (int i = 1; i < argc - 1; ++i) {
          if (std::string(argv[i]) == "--json")
            return argv[i + 1];
        }
        return {};
      }());
    } catch (const std::exception& write_error) {
      std::cerr << "failed to write JSON report: " << write_error.what() << "\n";
    }
    std::cerr << "fail: " << e.what() << "\n";
    return exit_code;
  }
}
