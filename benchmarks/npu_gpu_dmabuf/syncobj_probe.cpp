// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

#include <drm/amdgpu_drm.h>
#include <fcntl.h>
#include <libdrm/amdgpu.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>

#include "core/common/shim/hwctx_handle.h"
#include "shim/hwctx.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_uuid.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_xclbin.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kDefaultRenderNode = "/dev/dri/renderD128";
constexpr const char* kDefaultAccelNode = "/dev/accel/accel0";
constexpr const char* kDefaultXclbin =
  "/home/sadara/xdna-driver/build/vtd_extract/strx/validate_df_bandwidth.xclbin";
constexpr const char* kDefaultElf =
  "/home/sadara/xdna-driver/build/vtd_extract/strx/df_bw.elf";
constexpr const char* kDefaultKernel = "DPU";
constexpr uint64_t kDfBwProfileBufferBytes = 1024ull * 1024ull * 1024ull;

std::string
errno_string(int err)
{
  if (err < 0)
    err = -err;
  return std::strerror(err);
}

std::string
json_escape(const std::string& input)
{
  std::ostringstream os;
  for (unsigned char c : input) {
    switch (c) {
    case '"': os << "\\\""; break;
    case '\\': os << "\\\\"; break;
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

uint64_t
now_us()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(
           clock::now().time_since_epoch())
    .count();
}

int64_t
abs_timeout_ns(uint64_t timeout_ms)
{
  timespec ts = {};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed: " + errno_string(errno));
  return static_cast<int64_t>(ts.tv_sec) * 1000000000ll +
         static_cast<int64_t>(ts.tv_nsec) +
         static_cast<int64_t>(timeout_ms) * 1000000ll;
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

struct Options {
  std::string xclbin = kDefaultXclbin;
  std::string elf = kDefaultElf;
  std::string kernel = kDefaultKernel;
  std::string render_node = kDefaultRenderNode;
  std::string accel_node = kDefaultAccelNode;
  std::string json_path;
  int xrt_device = 0;
  uint64_t wait_timeout_ms = 60000;
};

void
usage(const char* argv0)
{
  std::cerr
    << "Usage: " << argv0 << " [options]\n"
    << "  --xclbin PATH\n"
    << "  --elf PATH\n"
    << "  --kernel NAME          default: DPU\n"
    << "  --render-node PATH     default: /dev/dri/renderD128\n"
    << "  --accel-node PATH      default: /dev/accel/accel0\n"
    << "  --xrt-device INDEX     default: 0\n"
    << "  --timeout-ms N         default: 60000\n"
    << "  --json PATH\n";
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

    if (key == "--xclbin") {
      opt.xclbin = need_value("--xclbin");
    } else if (key == "--elf") {
      opt.elf = need_value("--elf");
    } else if (key == "--kernel") {
      opt.kernel = need_value("--kernel");
    } else if (key == "--render-node") {
      opt.render_node = need_value("--render-node");
    } else if (key == "--accel-node") {
      opt.accel_node = need_value("--accel-node");
    } else if (key == "--xrt-device") {
      opt.xrt_device = std::stoi(need_value("--xrt-device"));
    } else if (key == "--timeout-ms") {
      opt.wait_timeout_ms = std::stoull(need_value("--timeout-ms"));
    } else if (key == "--json") {
      opt.json_path = need_value("--json");
    } else if (key == "-h" || key == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option '" + key + "'");
    }
  }
  return opt;
}

struct Stage {
  std::string name;
  std::string status;
  std::string detail;
  uint64_t us = 0;
};

struct Report {
  std::string result = "fail";
  std::string result_detail;
  uint32_t xdna_hwctx_handle = 0;
  uint32_t xdna_ctx_syncobj = 0;
  int xdna_ctx_fd = -1;
  uint64_t wait_point = 0;
  std::vector<Stage> stages;

  void
  add(std::string name, std::string status, std::string detail = {}, uint64_t us = 0)
  {
    stages.push_back({std::move(name), std::move(status), std::move(detail), us});
  }

  std::string
  to_json() const
  {
    std::ostringstream os;
    os << "{\n";
    os << "  \"result\": \"" << json_escape(result) << "\",\n";
    os << "  \"result_detail\": \"" << json_escape(result_detail) << "\",\n";
    os << "  \"xdna_hwctx_handle\": " << xdna_hwctx_handle << ",\n";
    os << "  \"xdna_ctx_syncobj\": " << xdna_ctx_syncobj << ",\n";
    os << "  \"xdna_ctx_fd\": " << xdna_ctx_fd << ",\n";
    os << "  \"wait_point\": " << wait_point << ",\n";
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
    if (ctx_)
      amdgpu_cs_ctx_free(ctx_);
    if (dev_)
      amdgpu_device_deinitialize(dev_);
  }

  amdgpu_device_handle
  dev() const
  {
    return dev_;
  }

  int
  fd() const
  {
    return fd_.get();
  }

  amdgpu_context_handle
  ctx()
  {
    if (!ctx_) {
      int ret = amdgpu_cs_ctx_create(dev_, &ctx_);
      if (ret)
        throw std::runtime_error("amdgpu_cs_ctx_create failed: " + errno_string(ret));
    }
    return ctx_;
  }

private:
  UniqueFd fd_;
  amdgpu_device_handle dev_ = nullptr;
  amdgpu_context_handle ctx_ = nullptr;
};

class AmdgpuBo {
public:
  AmdgpuBo(amdgpu_device_handle dev,
           uint64_t size,
           uint32_t domain = AMDGPU_GEM_DOMAIN_GTT,
           uint64_t flags = 0)
    : dev_(dev)
    , size_(size)
  {
    amdgpu_bo_alloc_request req = {};
    req.alloc_size = size_;
    req.phys_alignment = 4096;
    req.preferred_heap = domain;
    req.flags = flags;
    int ret = amdgpu_bo_alloc(dev_, &req, &bo_);
    if (ret)
      throw std::runtime_error("amdgpu_bo_alloc failed: " + errno_string(ret));
  }

  ~AmdgpuBo()
  {
    if (cpu_)
      amdgpu_bo_cpu_unmap(bo_);
    if (gpu_va_mapped_) {
      amdgpu_bo_va_op(bo_, 0, size_, gpu_va_, 0, AMDGPU_VA_OP_UNMAP);
      gpu_va_mapped_ = false;
    }
    if (va_handle_)
      amdgpu_va_range_free(va_handle_);
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

  uint32_t
  export_kms_handle() const
  {
    uint32_t kms_handle = 0;
    int ret = amdgpu_bo_export(bo_, amdgpu_bo_handle_type_kms, &kms_handle);
    if (ret)
      throw std::runtime_error("amdgpu_bo_export(kms) failed: " + errno_string(ret));
    return kms_handle;
  }

  uint64_t
  map_gpu_va()
  {
    if (gpu_va_)
      return gpu_va_;

    int ret = amdgpu_va_range_alloc(dev_,
                                    amdgpu_gpu_va_range_general,
                                    size_,
                                    4096,
                                    0,
                                    &gpu_va_,
                                    &va_handle_,
                                    0);
    if (ret)
      throw std::runtime_error("amdgpu_va_range_alloc failed: " + errno_string(ret));

    ret = amdgpu_bo_va_op(bo_, 0, size_, gpu_va_, 0, AMDGPU_VA_OP_MAP);
    if (ret)
      throw std::runtime_error("amdgpu_bo_va_op(MAP) failed: " + errno_string(ret));
    gpu_va_mapped_ = true;

    return gpu_va_;
  }

  void
  write_bytes(const void* src, size_t bytes)
  {
    if (bytes > size_)
      throw std::runtime_error("write exceeds amdgpu BO size");

    if (!cpu_) {
      void* ptr = nullptr;
      int ret = amdgpu_bo_cpu_map(bo_, &ptr);
      if (ret)
        throw std::runtime_error("amdgpu_bo_cpu_map failed: " + errno_string(ret));
      cpu_ = ptr;
    }

    std::memcpy(cpu_, src, bytes);
  }

private:
  amdgpu_device_handle dev_ = nullptr;
  uint64_t size_ = 0;
  amdgpu_bo_handle bo_ = nullptr;
  amdgpu_va_handle va_handle_ = nullptr;
  uint64_t gpu_va_ = 0;
  bool gpu_va_mapped_ = false;
  void* cpu_ = nullptr;
};

class AmdgpuRawBoList {
public:
  AmdgpuRawBoList(amdgpu_device_handle dev, const std::vector<AmdgpuBo*>& bos)
    : dev_(dev)
  {
    std::vector<drm_amdgpu_bo_list_entry> entries;
    entries.reserve(bos.size());
    for (const auto* bo : bos) {
      drm_amdgpu_bo_list_entry entry = {};
      entry.bo_handle = bo->export_kms_handle();
      entry.bo_priority = 0;
      entries.push_back(entry);
    }

    int ret = amdgpu_bo_list_create_raw(dev_,
                                        static_cast<uint32_t>(entries.size()),
                                        entries.data(),
                                        &handle_);
    if (ret)
      throw std::runtime_error("amdgpu_bo_list_create_raw failed: " + errno_string(ret));
  }

  ~AmdgpuRawBoList()
  {
    if (handle_)
      amdgpu_bo_list_destroy_raw(dev_, handle_);
  }

  AmdgpuRawBoList(const AmdgpuRawBoList&) = delete;
  AmdgpuRawBoList& operator=(const AmdgpuRawBoList&) = delete;

  uint32_t
  handle() const
  {
    return handle_;
  }

private:
  amdgpu_device_handle dev_ = nullptr;
  uint32_t handle_ = 0;
};

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

shim_xdna::hwctx*
get_xdna_hwctx(xrt::hw_context& hwctx)
{
  auto* base = static_cast<xrt_core::hwctx_handle*>(hwctx);
  auto* xdna = dynamic_cast<shim_xdna::hwctx*>(base);
  if (!xdna)
    throw std::runtime_error("xrt::hw_context is not backed by shim_xdna::hwctx");
  return xdna;
}

UniqueFd
export_syncobj_from_xrt_fd(uint32_t syncobj, const std::string& accel_node, int* owner_fd)
{
  struct stat accel_st = {};
  if (::stat(accel_node.c_str(), &accel_st) != 0)
    throw std::runtime_error("stat(" + accel_node + ") failed: " + errno_string(errno));

  DIR* raw_dir = ::opendir("/proc/self/fd");
  if (!raw_dir)
    throw std::runtime_error("opendir(/proc/self/fd) failed: " + errno_string(errno));
  std::unique_ptr<DIR, decltype(&::closedir)> dir(raw_dir, ::closedir);

  std::vector<std::string> attempts;
  while (dirent* ent = ::readdir(dir.get())) {
    if (ent->d_name[0] == '.')
      continue;

    char* end = nullptr;
    long fd_long = std::strtol(ent->d_name, &end, 10);
    if (!end || *end != '\0' || fd_long < 0)
      continue;
    int fd = static_cast<int>(fd_long);

    struct stat fd_st = {};
    if (::fstat(fd, &fd_st) != 0)
      continue;
    if (!S_ISCHR(fd_st.st_mode) || fd_st.st_rdev != accel_st.st_rdev)
      continue;

    int exported = -1;
    int ret = drmSyncobjHandleToFD(fd, syncobj, &exported);
    if (ret == 0) {
      if (owner_fd)
        *owner_fd = fd;
      return UniqueFd(exported);
    }

    std::ostringstream os;
    os << "fd " << fd << " export ret=" << ret << " errno=" << errno
       << " (" << errno_string(errno) << ")";
    attempts.push_back(os.str());
  }

  std::ostringstream os;
  os << "could not export XDNA syncobj " << syncobj << " from any open "
     << accel_node << " fd";
  if (!attempts.empty()) {
    os << "; attempts:";
    for (const auto& attempt : attempts)
      os << " [" << attempt << "]";
  }
  throw std::runtime_error(os.str());
}

uint32_t
import_syncobj_to_amdgpu(AmdgpuDevice& gpu, int shared_syncobj_fd)
{
  uint32_t handle = 0;
  int ret = drmSyncobjFDToHandle(gpu.fd(), shared_syncobj_fd, &handle);
  if (ret)
    throw std::runtime_error("drmSyncobjFDToHandle(amdgpu) failed: " + errno_string(errno));
  return handle;
}

void
wait_amdgpu_timeline(AmdgpuDevice& gpu, uint32_t handle, uint64_t point, uint64_t timeout_ms)
{
  uint32_t first = 0;
  int ret = drmSyncobjTimelineWait(gpu.fd(),
                                   &handle,
                                   &point,
                                   1,
                                   abs_timeout_ns(timeout_ms),
                                   DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
                                     DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
                                   &first);
  if (ret)
    throw std::runtime_error("drmSyncobjTimelineWait(amdgpu) failed: " + errno_string(errno));
}

bool
submit_amdgpu_wait_only(AmdgpuDevice& gpu,
                        uint32_t imported_syncobj,
                        uint64_t point,
                        std::string& detail,
                        uint64_t* seq_out)
{
  drm_amdgpu_cs_chunk_syncobj wait = {};
  wait.handle = imported_syncobj;
  wait.flags = 0;
  wait.point = point;

  drm_amdgpu_cs_chunk chunk = {};
  chunk.chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT;
  chunk.length_dw = sizeof(wait) / sizeof(uint32_t);
  chunk.chunk_data = reinterpret_cast<uint64_t>(&wait);

  uint64_t seq = 0;
  int ret = amdgpu_cs_submit_raw2(gpu.dev(), gpu.ctx(), 0, 1, &chunk, &seq);
  if (seq_out)
    *seq_out = seq;
  if (ret) {
    detail = "amdgpu_cs_submit_raw2(wait-only) failed: " + errno_string(ret);
    return false;
  }

  std::ostringstream os;
  os << "accepted wait-only CS, seq=" << seq;
  detail = os.str();
  return true;
}

bool
submit_amdgpu_nop_ib_wait(AmdgpuDevice& gpu,
                          uint32_t imported_syncobj,
                          uint64_t point,
                          uint64_t timeout_ms,
                          std::string& detail)
{
  constexpr uint32_t kPm4Nop[] = {
    (3u << 30) | (0x10u << 8) | 0u,
    0xdeadbeefu,
  };

  AmdgpuBo ib_bo(gpu.dev(), 4096, AMDGPU_GEM_DOMAIN_GTT, 0);
  ib_bo.write_bytes(kPm4Nop, sizeof(kPm4Nop));
  uint64_t ib_va = ib_bo.map_gpu_va();
  AmdgpuRawBoList bo_list(gpu.dev(), {&ib_bo});

  drm_amdgpu_cs_chunk_syncobj wait = {};
  wait.handle = imported_syncobj;
  wait.flags = 0;
  wait.point = point;

  drm_amdgpu_cs_chunk_ib ib = {};
  ib.flags = 0;
  ib.va_start = ib_va;
  ib.ib_bytes = sizeof(kPm4Nop);
  ib.ip_type = AMDGPU_HW_IP_COMPUTE;
  ib.ip_instance = 0;
  ib.ring = 0;

  drm_amdgpu_cs_chunk chunks[2] = {};
  chunks[0].chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT;
  chunks[0].length_dw = sizeof(wait) / sizeof(uint32_t);
  chunks[0].chunk_data = reinterpret_cast<uint64_t>(&wait);
  chunks[1].chunk_id = AMDGPU_CHUNK_ID_IB;
  chunks[1].length_dw = sizeof(ib) / sizeof(uint32_t);
  chunks[1].chunk_data = reinterpret_cast<uint64_t>(&ib);

  uint64_t seq = 0;
  int ret = amdgpu_cs_submit_raw2(gpu.dev(), gpu.ctx(), bo_list.handle(), 2, chunks, &seq);
  if (ret) {
    detail = "amdgpu_cs_submit_raw2(wait+NOP-IB) failed: " + errno_string(ret);
    return false;
  }

  amdgpu_cs_fence fence = {};
  fence.context = gpu.ctx();
  fence.ip_type = AMDGPU_HW_IP_COMPUTE;
  fence.ip_instance = 0;
  fence.ring = 0;
  fence.fence = seq;

  uint32_t expired = 0;
  ret = amdgpu_cs_query_fence_status(&fence,
                                     abs_timeout_ns(timeout_ms),
                                     AMDGPU_QUERY_FENCE_TIMEOUT_IS_ABSOLUTE,
                                     &expired);
  if (ret) {
    detail = "amdgpu_cs_query_fence_status failed after accepted CS seq=" +
             std::to_string(seq) + ": " + errno_string(ret);
    return false;
  }
  if (!expired) {
    detail = "wait+NOP-IB CS timed out, seq=" + std::to_string(seq);
    return false;
  }

  detail = "accepted and completed wait+NOP-IB CS on compute ring, seq=" +
           std::to_string(seq);
  return true;
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

  try {
    Options opt = parse_args(argc, argv);

    uint64_t t0 = now_us();
    XrtContext xrt(opt);
    report.add("xrt_context", "pass", {}, now_us() - t0);

    auto* xdna_hwctx = get_xdna_hwctx(xrt.hwctx);
    report.xdna_hwctx_handle = xdna_hwctx->get_slotidx();
    report.xdna_ctx_syncobj = xdna_hwctx->get_syncobj();
    report.wait_point = 0;

    t0 = now_us();
    int owner_fd = -1;
    UniqueFd xdna_syncobj_fd =
      export_syncobj_from_xrt_fd(report.xdna_ctx_syncobj, opt.accel_node, &owner_fd);
    report.xdna_ctx_fd = owner_fd;
    report.add("export_xdna_context_syncobj", "pass", "shared syncobj fd exported", now_us() - t0);

    t0 = now_us();
    AmdgpuDevice gpu(opt.render_node);
    report.add("amdgpu_open", "pass", {}, now_us() - t0);

    t0 = now_us();
    uint32_t amdgpu_syncobj = import_syncobj_to_amdgpu(gpu, xdna_syncobj_fd.get());
    report.add("amdgpu_import_xdna_syncobj", "pass",
               "imported handle=" + std::to_string(amdgpu_syncobj), now_us() - t0);

    AmdgpuBo output_bo(gpu.dev(), kDfBwProfileBufferBytes);
    UniqueFd output_fd = output_bo.export_dmabuf();
    xrt::bo output(xrt.device, static_cast<xrt::bo::export_handle>(output_fd.get()));
    xrt::bo input(xrt.device, kDfBwProfileBufferBytes, xrt::bo::flags::host_only, 0);

    xrt::run run(xrt.kernel);
    set_df_bw_args(run, input, output);

    t0 = now_us();
    run.start();
    report.add("npu_start", "pass", {}, now_us() - t0);

    t0 = now_us();
    std::string cs_detail;
    uint64_t wait_only_seq = 0;
    bool wait_only_ok = submit_amdgpu_wait_only(gpu,
                                                amdgpu_syncobj,
                                                report.wait_point,
                                                cs_detail,
                                                &wait_only_seq);
    report.add("amdgpu_cs_wait_only", wait_only_ok ? "pass" : "fail", cs_detail, now_us() - t0);

    t0 = now_us();
    bool nop_ib_ok = submit_amdgpu_nop_ib_wait(gpu,
                                               amdgpu_syncobj,
                                               report.wait_point,
                                               opt.wait_timeout_ms,
                                               cs_detail);
    report.add("amdgpu_cs_wait_nop_ib", nop_ib_ok ? "pass" : "fail", cs_detail, now_us() - t0);

    t0 = now_us();
    wait_amdgpu_timeline(gpu, amdgpu_syncobj, report.wait_point, opt.wait_timeout_ms);
    report.add("amdgpu_cpu_timeline_wait", "pass", "amdgpu waited on imported XDNA timeline point", now_us() - t0);

    auto state = run.wait(opt.wait_timeout_ms);
    if (state != ERT_CMD_STATE_COMPLETED) {
      std::ostringstream os;
      os << "xrt run.wait returned state " << static_cast<int>(state);
      throw std::runtime_error(os.str());
    }
    report.add("xrt_run_wait_after_probe", "pass");

    if (nop_ib_ok) {
      report.result = "pass";
      report.result_detail =
        "XDNA context syncobj exported and imported by amdgpu; amdgpu CS timeline wait completed without a polling bridge";
    } else if (wait_only_ok) {
      report.result = "partial";
      report.result_detail =
        "XDNA context syncobj imported by amdgpu and wait-only CS was accepted; NOP IB completion probe failed";
    } else {
      report.result = "partial";
      report.result_detail =
        "XDNA context syncobj imported by amdgpu and CPU timeline wait passed; amdgpu CS wait submission failed";
    }

    write_report(report, opt.json_path);
    std::cout << report.to_json();
    return report.result == "pass" ? 0 : 2;
  } catch (const std::exception& e) {
    report.result = "fail";
    report.result_detail = e.what();
    report.add("fatal", "fail", e.what());
    try {
      if (argc > 1) {
        Options opt = parse_args(argc, argv);
        write_report(report, opt.json_path);
      }
    } catch (...) {
    }
    std::cerr << "error: " << e.what() << "\n";
    std::cout << report.to_json();
    return 1;
  }
}
