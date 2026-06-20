#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static int check_hip(const char* label, hipError_t err) {
    if (err == hipSuccess) {
        std::printf("%s OK\n", label);
        return 0;
    }
    std::fprintf(stderr, "%s failed: %s (%d)\n", label, hipGetErrorString(err),
                 static_cast<int>(err));
    return 5;
}

static bool mode_is(const char* mode, const char* expected) {
    return std::strcmp(mode, expected) == 0;
}

static int maybe_parent_hip_init(const char* mode) {
    if (mode_is(mode, "plain")) {
        return 0;
    }

    int count = 0;
    int rc = check_hip("parent hipGetDeviceCount", hipGetDeviceCount(&count));
    if (rc != 0) {
        return rc;
    }
    std::printf("parent device_count=%d\n", count);

    rc = check_hip("parent hipSetDevice(0)", hipSetDevice(0));
    if (rc != 0) {
        return rc;
    }

    rc = check_hip("parent hipDeviceSynchronize", hipDeviceSynchronize());
    if (rc != 0) {
        return rc;
    }

    if (mode_is(mode, "hipinit_reset_before")) {
        return check_hip("parent hipDeviceReset before children",
                         hipDeviceReset());
    }

    if (mode_is(mode, "hipinit") || mode_is(mode, "hipinit_reset_between")) {
        return 0;
    }

    std::fprintf(stderr,
                 "unknown mode '%s' (use plain, hipinit, "
                 "hipinit_reset_before, hipinit_reset_between)\n",
                 mode);
    return 2;
}

static std::vector<int> parse_chunks(const char* text) {
    std::vector<int> chunks;
    std::stringstream ss(text ? text : "");
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        chunks.push_back(std::atoi(token.c_str()));
    }
    return chunks;
}

static int run_child(const char* label, const char* phase_bin, int launches,
                     int grid_x, int grid_y) {
    std::printf("parent launching %s child launches=%d grid=%dx%d\n", label,
                launches, grid_x, grid_y);
    std::fflush(stdout);
    std::fflush(stderr);

    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork %s failed: %s\n", label, std::strerror(errno));
        return 6;
    }

    if (pid == 0) {
        std::string launches_arg = std::to_string(launches);
        std::string grid_x_arg = std::to_string(grid_x);
        std::string grid_y_arg = std::to_string(grid_y);
        char* const argv[] = {
            const_cast<char*>(phase_bin),
            const_cast<char*>(launches_arg.c_str()),
            const_cast<char*>("0"),
            const_cast<char*>(grid_x_arg.c_str()),
            const_cast<char*>(grid_y_arg.c_str()),
            const_cast<char*>("same"),
            nullptr,
        };
        execv(phase_bin, argv);
        std::fprintf(stderr, "execv %s failed: %s\n", phase_bin,
                     std::strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "waitpid %s failed: %s\n", label,
                     std::strerror(errno));
        return 7;
    }

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        std::printf("child %s exited %d\n", label, code);
        return code;
    }

    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        std::fprintf(stderr, "child %s signaled %d\n", label, sig);
        return 128 + sig;
    }

    std::fprintf(stderr, "child %s ended unexpectedly, status=%d\n", label,
                 status);
    return 8;
}

int main(int argc, char** argv) {
    const char* phase_bin = argc > 1 ? argv[1] : nullptr;
    const char* chunks_text = argc > 2 ? argv[2] : "96,5";
    int grid_x = argc > 3 ? std::atoi(argv[3]) : 512;
    int grid_y = argc > 4 ? std::atoi(argv[4]) : 86;
    const char* mode = argc > 5 ? argv[5] : "plain";

    if (phase_bin == nullptr || phase_bin[0] == '\0') {
        std::fprintf(stderr,
                     "usage: %s <phase_probe_bin> [chunk_csv grid_x grid_y "
                     "mode]\n",
                     argv[0]);
        return 2;
    }

    std::vector<int> chunks = parse_chunks(chunks_text);
    if (chunks.empty()) {
        std::fprintf(stderr, "no chunks supplied\n");
        return 2;
    }

    int total = 0;
    for (int chunk : chunks) {
        if (chunk < 0) {
            std::fprintf(stderr, "negative chunk %d is invalid\n", chunk);
            return 2;
        }
        total += chunk;
    }

    std::printf("multi-exec-parent mode=%s phase_bin=%s chunks=%s total=%d "
                "grid=%dx%d parent_pid=%ld\n",
                mode, phase_bin, chunks_text, total, grid_x, grid_y,
                static_cast<long>(getpid()));

    int rc = maybe_parent_hip_init(mode);
    if (rc != 0) {
        return rc;
    }

    for (size_t i = 0; i < chunks.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "chunk%zu", i);
        rc = run_child(label, phase_bin, chunks[i], grid_x, grid_y);
        if (rc != 0) {
            return rc;
        }

        if (mode_is(mode, "hipinit_reset_between") && i + 1 < chunks.size()) {
            rc = check_hip("parent hipDeviceReset between children",
                           hipDeviceReset());
            if (rc != 0) {
                return rc;
            }
        }
    }

    std::printf("multi-exec-parent mode=%s chunks=%s total=%d OK\n", mode,
                chunks_text, total);
    return 0;
}
