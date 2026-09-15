#include "State_Mimic.h"
#include "State_Walk.h"
#include <mujoco/mujoco.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

std::unique_ptr<LowCmd_t> FSMState::lowcmd;
std::shared_ptr<LowState_t> FSMState::lowstate;
std::shared_ptr<Keyboard> FSMState::keyboard;

int main(int argc,char**argv) {
 if(argc<5) { std::cerr<<"sim scene.xml output.csv jump|walk blend [seconds]\n"; return 2; }
 param::proj_dir="/home/ustczxh/humanoid/zxh-isaaclab/deploy/robots/g1_29dof";
 param::config=YAML::LoadFile(param::proj_dir/"config/config.yaml");
 param::config["FSM"]["Velocity"]["walk_entry"]["blend_time_s"]=std::stod(argv[4]);
 for(const auto&item:param::config["FSM"]["_"]) FSMStringMap.insert({item.second["id"].as<int>(),item.first.as<std::string>()});
 FSMState::lowcmd=std::make_unique<LowCmd_t>(); FSMState::lowstate=std::make_shared<LowState_t>();
 FSMState::control_observer=State_Walk::record_control_frame;
 char err[4096]; mjModel*m=mj_loadXML(argv[1],nullptr,err,sizeof(err));
 if(!m) {std::cerr<<err;return 2;} m->opt.timestep=.001;
 mjData*d=mj_makeData(m);
 int qa[29],va[29];
 for(int i=0;i<29;++i) {int j=m->actuator_trnid[2*i]; qa[i]=m->jnt_qposadr[j];va[i]=m->jnt_dofadr[j];}
 bool jump=std::string(argv[3])=="jump";
 bool sequence=std::string(argv[3])=="sequence";
 if(jump) {
  std::ifstream file(param::proj_dir/"config/policy/mimic/jump3/params/jump3_waist15.csv");
  std::string line; std::getline(file,line); std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line); double q[36];for(auto&v:q)row>>v;
  for(int i=0;i<3;++i)d->qpos[i]=q[i];
  d->qpos[3]=q[6];for(int i=0;i<3;++i)d->qpos[4+i]=q[3+i];
  for(int i=0;i<29;++i)d->qpos[qa[i]]=q[7+i];
 } else {
  auto cfg=YAML::LoadFile(param::proj_dir/"config/policy/velocity/params/deploy.yaml");
  auto ids=cfg["joint_ids_map"].as<std::vector<int>>();auto q=cfg["default_joint_pos"].as<std::vector<float>>();
  for(int i=0;i<29;++i)d->qpos[qa[ids[i]]]=q[i]; d->qpos[2]=.78;
 }
 mj_forward(m,d);
 int iq=m->sensor_adr[mj_name2id(m,mjOBJ_SENSOR,"imu_quat")],ig=m->sensor_adr[mj_name2id(m,mjOBJ_SENSOR,"imu_gyro")];
 int torso=mj_name2id(m,mjOBJ_BODY,"torso_link");if(torso<0) throw std::runtime_error("Missing torso");
 auto sync=[&]{std::lock_guard<std::mutex>lock(FSMState::lowstate->mutex_);auto&s=FSMState::lowstate->msg_;for(int i=0;i<29;++i){s.motors[i].q()=d->qpos[qa[i]];s.motors[i].dq()=d->qvel[va[i]];}for(int i=0;i<4;++i)s.imu.quat[i]=d->sensordata[iq+i];for(int i=0;i<3;++i)s.imu.gyro[i]=d->sensordata[ig+i];};
 sync();for(int i=0;i<29;++i)FSMState::lowcmd->msg_.motors[i].q()=d->qpos[qa[i]];
 State_Mimic mimic(112,"Mimic_Jump3");State_Walk walk(3,"Velocity");
 for(int i=0;i<90;++i){State_Walk::record_control_frame();std::this_thread::sleep_for(std::chrono::milliseconds(1));}
 FSMState*state=jump?static_cast<FSMState*>(&mimic):static_cast<FSMState*>(&walk);
 std::ofstream out(argv[2]);out<<std::setprecision(9)<<"time,state,z,vx,vy,vz,pelvis_tilt,torso_tilt,contacts";
 for(int i=0;i<m->nq;++i)out<<",qpos"<<i;
 for(int i=0;i<m->nv;++i)out<<",qvel"<<i;
 for(int i=0;i<29;++i)out<<",target"<<i;
 for(int i=0;i<29;++i)out<<",torque"<<i;
 out<<'\n';
 state->enter();using Clock=std::chrono::steady_clock;auto start=Clock::now();
 int sid=jump?112:3;double sw=-1,passive=-1;double seconds=argc>5?std::stod(argv[5]):6;
 double maxlag=0;
 for(int tick=0;tick<seconds*1000;++tick){
  sync();
  if(sequence && tick==2000 && state==&walk){
   state->exit();state=&mimic;sid=112;
   std::cout<<"TRIGGER walk -> jump at "<<d->time<<std::endl;state->enter();
  }
  if(state){state->pre_run();state->run();state->post_run();
   for(auto&check:state->registered_checks)if(check.first()){
    int next=check.second;state->exit();
    std::cout<<"TRANSITION "<<sid<<" -> "<<next<<" sim_t="<<d->time<<" z="<<d->qpos[2]<<std::endl;
    if(next==3){
     // Optional one-time body-forward velocity / body-pitch angular-velocity impulse.
     double dv=argc>6?std::stod(argv[6]):0, dw=argc>7?std::stod(argv[7]):0;
     double yaw=std::atan2(2*(d->qpos[3]*d->qpos[6]+d->qpos[4]*d->qpos[5]),1-2*(d->qpos[5]*d->qpos[5]+d->qpos[6]*d->qpos[6]));
     d->qvel[0]+=dv*std::cos(yaw);d->qvel[1]+=dv*std::sin(yaw);d->qvel[4]+=dw;
     mj_forward(m,d);sync();state=&walk;sw=d->time;state->enter();
    }
    else{state=nullptr;passive=d->time;}sid=next;break;
   }
  }
  for(int i=0;i<29;++i){auto&c=FSMState::lowcmd->msg_.motors[i];d->ctrl[i]=state?(c.tau()+c.kp()*(c.q()-d->qpos[qa[i]])+c.kd()*(c.dq()-d->qvel[va[i]])):-3*d->qvel[va[i]];}
  double tilt=std::acos(std::clamp(1-2*(d->qpos[4]*d->qpos[4]+d->qpos[5]*d->qpos[5]),-1.,1.));
  double tt=std::acos(std::clamp(d->xmat[9*torso+8],-1.,1.));
  out<<d->time<<','<<sid<<','<<d->qpos[2]<<','<<d->qvel[0]<<','<<d->qvel[1]<<','<<d->qvel[2]<<','<<tilt<<','<<tt<<','<<d->ncon;
  for(int i=0;i<m->nq;++i)out<<','<<d->qpos[i];for(int i=0;i<m->nv;++i)out<<','<<d->qvel[i];
  for(int i=0;i<29;++i)out<<','<<FSMState::lowcmd->msg_.motors[i].q();for(int i=0;i<29;++i)out<<','<<d->ctrl[i];out<<'\n';
  mj_step(m,d);
  auto deadline=start+std::chrono::microseconds((tick+1)*1000);maxlag=std::max(maxlag,std::chrono::duration<double>(Clock::now()-deadline).count());std::this_thread::sleep_until(deadline);
  if(passive>=0 && d->time>passive+1)break;
 }
 if(state)state->exit();out.close();
 std::cout<<"DONE switch="<<sw<<" passive="<<passive<<" final_z="<<d->qpos[2]<<" max_wall_lag="<<maxlag<<std::endl;
 mj_deleteData(d);mj_deleteModel(m);
}
