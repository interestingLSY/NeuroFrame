#pragma once

#include <cstdio>
#include <cstdlib>

namespace NeuroFrame {

#define LOG(level_str, fmtstr, ...) \
	fprintf(stderr, "[%s] (%s:%d) " fmtstr "\n", level_str, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_FATAL(fmtstr, ...) \
	{LOG("FATAL", fmtstr, ##__VA_ARGS__); \
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
