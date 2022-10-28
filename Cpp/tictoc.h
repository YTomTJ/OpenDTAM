#ifndef TIC_TOC
#define TIC_TOC

#include <iostream>
#include <ctime>
#include <boost/date_time.hpp>

static double wall_time()
{
    static const boost::posix_time::ptime t0(boost::gregorian::date(1970, 1, 1));
    const boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
    const boost::posix_time::time_duration dt(t1 - t0);
    return dt.total_microseconds() / 1e6;
}

// Just a static variable that will be visible only within this file

static double start_time = 0;
static void tic() { start_time = wall_time(); }

static double toc()
{
    double elapsed = wall_time() - start_time;
    std::cout << "Elapsed time is " << elapsed << " seconds." << std::endl;
    return elapsed;
}

static double tocq() { return wall_time() - start_time; }

#endif // TIC_TOC
