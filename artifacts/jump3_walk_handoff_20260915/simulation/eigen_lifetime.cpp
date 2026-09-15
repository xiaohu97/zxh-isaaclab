#include <Eigen/Dense>
#include <iostream>
int main(int argc,char**) {
 Eigen::Quaternionf q(Eigen::AngleAxisf(0.2f,Eigen::Vector3f::UnitY()));
 if(argc==1) {
  auto rot=q.toRotationMatrix().transpose();
  Eigen::Matrix<float,6,1> obs;
  obs << rot(0,0),rot(0,1),rot(1,0),rot(1,1),rot(2,0),rot(2,1);
  std::cout<<obs.transpose()<<'\n';
 } else {
  const Eigen::Matrix3f rot=q.toRotationMatrix().transpose();
  Eigen::Matrix<float,6,1> obs;
  obs << rot(0,0),rot(0,1),rot(1,0),rot(1,1),rot(2,0),rot(2,1);
  std::cout<<obs.transpose()<<'\n';
 }
}
