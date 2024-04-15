#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>

#include <errno.h>

// Get the values from these macros because calling them from rust is not possible
const int sc_libav_averror_eof = AVERROR_EOF;
const int sc_libav_averror_eagain = AVERROR(EAGAIN);
