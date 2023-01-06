export module SE.Core.Log;

export import :LogStream;
export import :Logger;
export import :LogManager;

/**
* ----------------------------------------
* EXAMPLES
* ----------------------------------------
* Users should call predefined static functions in LogManager to log a message.
* Pass a string to logger, construct it with std::format if necessary.
* Logger will attach a timestamp in the beginning and change the color according to log type.
* 
* ```
* Core::LogManager::Debug(some_string);		\\ output a debug-type	 log, use cyan color
* Core::LogManager::Log(some_string);		\\ output a log-type	 log, use white color
* Core::LogManager::Warning(some_string);	\\ output a warning-type log, use yellow color
* Core::LogManager::Error(some_string);		\\ output a error-type	 log, use red color
* ```
*/