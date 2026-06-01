#pragma once

#include "rclcpp/rclcpp.hpp"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace incremental_dude_ros2::logging {

enum class RosLogLevel {
  Debug,
  Info,
  Warn,
  Error,
  Fatal,
};

inline rclcpp::Logger dudeThirdPartyLogger() {
  return rclcpp::get_logger("incremental_dude.third_party");
}

inline rclcpp::Clock &dudeThirdPartyClock() {
  static rclcpp::Clock clock(RCL_STEADY_TIME);
  return clock;
}

inline std::string trimMessage(std::string message) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  const auto begin = std::find_if(message.begin(), message.end(), not_space);
  if (begin == message.end()) {
    return {};
  }
  const auto end =
      std::find_if(message.rbegin(), message.rend(), not_space).base();
  return std::string(begin, end);
}

inline void emitMessage(RosLogLevel level, const std::string &message) {
  const std::string trimmed = trimMessage(message);
  if (trimmed.empty()) {
    return;
  }

  auto logger = dudeThirdPartyLogger();
  switch (level) {
  case RosLogLevel::Debug:
    RCLCPP_DEBUG(logger, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Info:
    RCLCPP_INFO(logger, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Warn:
    RCLCPP_WARN(logger, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Error:
    RCLCPP_ERROR(logger, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Fatal:
    RCLCPP_FATAL(logger, "%s", trimmed.c_str());
    break;
  }
}

inline void emitMessageThrottle(RosLogLevel level, int duration_ms,
                                const std::string &message) {
  const std::string trimmed = trimMessage(message);
  if (trimmed.empty()) {
    return;
  }

  auto logger = dudeThirdPartyLogger();
  auto &clock = dudeThirdPartyClock();
  switch (level) {
  case RosLogLevel::Debug:
    RCLCPP_DEBUG_THROTTLE(logger, clock, duration_ms, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Info:
    RCLCPP_INFO_THROTTLE(logger, clock, duration_ms, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Warn:
    RCLCPP_WARN_THROTTLE(logger, clock, duration_ms, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Error:
    RCLCPP_ERROR_THROTTLE(logger, clock, duration_ms, "%s", trimmed.c_str());
    break;
  case RosLogLevel::Fatal:
    RCLCPP_FATAL_THROTTLE(logger, clock, duration_ms, "%s", trimmed.c_str());
    break;
  }
}

class RosLogStream {
public:
  explicit RosLogStream(RosLogLevel level) : level_(level) {}
  RosLogStream(const RosLogStream &) = delete;
  RosLogStream &operator=(const RosLogStream &) = delete;

  RosLogStream(RosLogStream &&other) noexcept
      : level_(other.level_), stream_(std::move(other.stream_)),
        flushed_(other.flushed_) {
    other.flushed_ = true;
  }

  ~RosLogStream() { flush(); }

  template <typename T>
  RosLogStream &operator<<(const T &value) {
    stream_ << value;
    return *this;
  }

  RosLogStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
    manip(stream_);
    return *this;
  }

  RosLogStream &operator<<(std::ios_base &(*manip)(std::ios_base &)) {
    manip(stream_);
    return *this;
  }

  void flush() {
    if (flushed_) {
      return;
    }
    flushed_ = true;
    emitMessage(level_, stream_.str());
  }

private:
  RosLogLevel level_;
  std::ostringstream stream_;
  bool flushed_{false};
};

inline std::string vformatMessage(const char *format, va_list args) {
  if (format == nullptr) {
    return {};
  }

  va_list size_args;
  va_copy(size_args, args);
  const int size = std::vsnprintf(nullptr, 0, format, size_args);
  va_end(size_args);
  if (size <= 0) {
    return format;
  }

  std::vector<char> buffer(static_cast<size_t>(size) + 1U, '\0');
  std::vsnprintf(buffer.data(), buffer.size(), format, args);
  return std::string(buffer.data(), static_cast<size_t>(size));
}

inline void logf(RosLogLevel level, const char *format, ...) {
  va_list args;
  va_start(args, format);
  const std::string message = vformatMessage(format, args);
  va_end(args);
  emitMessage(level, message);
}

inline void logfThrottle(RosLogLevel level, int duration_ms, const char *format,
                         ...) {
  va_list args;
  va_start(args, format);
  const std::string message = vformatMessage(format, args);
  va_end(args);
  emitMessageThrottle(level, duration_ms, message);
}

} // namespace incremental_dude_ros2::logging

#define DUDE_ROS_DEBUG_STREAM()                                              \
  ::incremental_dude_ros2::logging::RosLogStream(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Debug)
#define DUDE_ROS_INFO_STREAM()                                               \
  ::incremental_dude_ros2::logging::RosLogStream(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Info)
#define DUDE_ROS_WARN_STREAM()                                               \
  ::incremental_dude_ros2::logging::RosLogStream(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Warn)
#define DUDE_ROS_ERROR_STREAM()                                              \
  ::incremental_dude_ros2::logging::RosLogStream(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Error)
#define DUDE_ROS_FATAL_STREAM()                                              \
  ::incremental_dude_ros2::logging::RosLogStream(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Fatal)

#define DUDE_ROS_DEBUGF(...)                                                 \
  ::incremental_dude_ros2::logging::logf(                                   \
      ::incremental_dude_ros2::logging::RosLogLevel::Debug, __VA_ARGS__)
#define DUDE_ROS_INFOF(...)                                                  \
  ::incremental_dude_ros2::logging::logf(                                   \
      ::incremental_dude_ros2::logging::RosLogLevel::Info, __VA_ARGS__)
#define DUDE_ROS_WARNF(...)                                                  \
  ::incremental_dude_ros2::logging::logf(                                   \
      ::incremental_dude_ros2::logging::RosLogLevel::Warn, __VA_ARGS__)
#define DUDE_ROS_ERRORF(...)                                                 \
  ::incremental_dude_ros2::logging::logf(                                   \
      ::incremental_dude_ros2::logging::RosLogLevel::Error, __VA_ARGS__)
#define DUDE_ROS_FATALF(...)                                                 \
  ::incremental_dude_ros2::logging::logf(                                   \
      ::incremental_dude_ros2::logging::RosLogLevel::Fatal, __VA_ARGS__)

#define DUDE_ROS_DEBUGF_THROTTLE(duration_ms, ...)                           \
  ::incremental_dude_ros2::logging::logfThrottle(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Debug, duration_ms,    \
      __VA_ARGS__)
#define DUDE_ROS_INFOF_THROTTLE(duration_ms, ...)                            \
  ::incremental_dude_ros2::logging::logfThrottle(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Info, duration_ms,     \
      __VA_ARGS__)
#define DUDE_ROS_WARNF_THROTTLE(duration_ms, ...)                            \
  ::incremental_dude_ros2::logging::logfThrottle(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Warn, duration_ms,     \
      __VA_ARGS__)
#define DUDE_ROS_ERRORF_THROTTLE(duration_ms, ...)                           \
  ::incremental_dude_ros2::logging::logfThrottle(                           \
      ::incremental_dude_ros2::logging::RosLogLevel::Error, duration_ms,    \
      __VA_ARGS__)
