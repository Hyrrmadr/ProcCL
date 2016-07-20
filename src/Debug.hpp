#ifndef DEBUG_H

#ifdef __APPLE__
# include <execinfo.h>
# include <stdio.h>
#endif
#include <iostream>

static void printStacktrace()
{
  std::cerr << "Stacktrace:" << std::endl;

#ifdef __APPLE__
  int pointersNumber = 20;
  void* callstack[pointersNumber];
  int frames = backtrace(callstack, pointersNumber);

  char** strs = backtrace_symbols(callstack, frames);
  for (int i = 0; i < frames; ++i)
    std::cerr << "\t" << strs[i] << std::endl;
  free(strs);
#endif
}
#endif
