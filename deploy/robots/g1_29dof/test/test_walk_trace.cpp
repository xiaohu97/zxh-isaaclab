#include "walk_trace.h"
#include <iostream>
int main(){
 const auto id=std::chrono::steady_clock::now().time_since_epoch().count();
 auto dir=std::filesystem::temp_directory_path()/("g1-walk-trace-test-"+std::to_string(id));
 {
  g1::WalkTrace trace(dir);g1::WalkTraceRow row;row.event=1;trace.push(row);
  for(int i=0;i<200;++i){row.event=0;row.elapsed=i*.01;row.frame.q.fill(i);row.applied.fill(i+.5f);trace.push(row);}
  row.event=3;row.reason=64;trace.push(row);row.event=2;trace.push(row);
  if(trace.dropped())throw std::runtime_error("A complete two-second trace did not fit the queue");
 }
 int files=0,lines=0;bool fault=false;
 for(const auto&f:std::filesystem::directory_iterator(dir)){
  ++files;std::ifstream in(f.path());std::string line;
  while(std::getline(in,line)){++lines;if(std::count(line.begin(),line.end(),',')!=164)throw std::runtime_error("Incomplete CSV row");if(line.find(",3,64,")!=std::string::npos)fault=true;}
 }
 std::filesystem::remove_all(dir);
 if(files!=1||lines!=204||!fault)throw std::runtime_error("Trace did not preserve samples and fault event on shutdown");
 std::cout<<"PASS bounded asynchronous trace queue, complete CSV columns, fault event and shutdown flush\n";
}
