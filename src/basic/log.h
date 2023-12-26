#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace NeuroFrame {

#define LOG(level_str, fmtstr, ...) \
	fprintf(stderr, "[%s] (%s:%d) " fmtstr "\n", level_str, __FILE__, __LINE__, ##__VA_ARGS__)

namespace Logging {
	extern const int LOG_FATAL_BUF_SIZE;
	extern char* log_fatal_buf;
}

#define LOG_FATAL(fmtstr, ...) \
	{LOG("FATAL", fmtstr, ##__VA_ARGS__); \
	 snprintf(Logging::log_fatal_buf, Logging::LOG_FATAL_BUF_SIZE, fmtstr, ##__VA_ARGS__); \
	 throw std::runtime_error(Logging::log_fatal_buf); \
	 exit(1); }

#define LOG_ERROR(fmtstr, ...) \
	LOG("ERROR", fmtstr, ##__VA_ARGS__)

#define LOG_WARN(fmtstr, ...) \
	LOG("WARN", fmtstr, ##__VA_ARGS__)

#define LOG_INFO(fmtstr, ...) \
	LOG("INFO", fmtstr, ##__VA_ARGS__)

#define LOG_DEBUG(fmtstr, ...) \
	LOG("DEBUG", fmtstr, ##__VA_ARGS__)

void print_cuda_error();

}
