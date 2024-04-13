#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>

#include <errno.h>

const int sc_libav_averror_eof = AVERROR_EOF;
const int sc_libav_averror_eagain = AVERROR(EAGAIN);
