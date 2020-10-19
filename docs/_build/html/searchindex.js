Search.setIndex({docnames:["guide/getting_started","guide/install","index","modules/blackbox_mpc","modules/dynamics_functions/dynamics_functions","modules/dynamics_handlers/dynamics_handlers","modules/environment_utils/environment_utils","modules/optimizers/optimizers","modules/policies/policies","modules/trajectory_evaluators/trajectory_evaluators","modules/utils/utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["guide/getting_started.rst","guide/install.rst","index.rst","modules/blackbox_mpc.rst","modules/dynamics_functions/dynamics_functions.rst","modules/dynamics_handlers/dynamics_handlers.rst","modules/environment_utils/environment_utils.rst","modules/optimizers/optimizers.rst","modules/policies/policies.rst","modules/trajectory_evaluators/trajectory_evaluators.rst","modules/utils/utils.rst"],objects:{"blackbox_mpc.dynamics_functions":{DeterministicMLP:[4,1,1,""]},"blackbox_mpc.dynamics_functions.DeterministicMLP":{__call__:[4,2,1,""],__init__:[4,2,1,""],get_loss:[4,2,1,""],get_validation_loss:[4,2,1,""]},"blackbox_mpc.dynamics_handlers":{SystemDynamicsHandler:[5,1,1,""]},"blackbox_mpc.dynamics_handlers.SystemDynamicsHandler":{__init__:[5,2,1,""],__weakref__:[5,3,1,""],get_dynamics_function:[5,2,1,""],process_input:[5,2,1,""],process_output:[5,2,1,""],train:[5,2,1,""]},"blackbox_mpc.environment_utils":{EnvironmentWrapper:[6,1,1,""]},"blackbox_mpc.environment_utils.EnvironmentWrapper":{__weakref__:[6,3,1,""],make_custom_gym_env:[6,2,1,""],make_standard_gym_env:[6,2,1,""]},"blackbox_mpc.optimizers":{CEMOptimizer:[7,1,1,""],CMAESOptimizer:[7,1,1,""],OptimizerBase:[7,1,1,""],PI2Optimizer:[7,1,1,""],PSOOptimizer:[7,1,1,""],RandomSearchOptimizer:[7,1,1,""],SPSAOptimizer:[7,1,1,""]},"blackbox_mpc.optimizers.CEMOptimizer":{__init__:[7,2,1,""],reset:[7,2,1,""]},"blackbox_mpc.optimizers.CMAESOptimizer":{__init__:[7,2,1,""],reset:[7,2,1,""]},"blackbox_mpc.optimizers.OptimizerBase":{__call__:[7,2,1,""],__init__:[7,2,1,""],reset:[7,2,1,""],set_trajectory_evaluator:[7,2,1,""]},"blackbox_mpc.optimizers.PI2Optimizer":{__init__:[7,2,1,""],reset:[7,2,1,""]},"blackbox_mpc.optimizers.PSOOptimizer":{__init__:[7,2,1,""],reset:[7,2,1,""]},"blackbox_mpc.optimizers.RandomSearchOptimizer":{__init__:[7,2,1,""],reset:[7,2,1,""]},"blackbox_mpc.optimizers.SPSAOptimizer":{__init__:[7,2,1,""],reset:[7,2,1,""]},"blackbox_mpc.policies":{MPCPolicy:[8,1,1,""],ModelBasedBasePolicy:[8,1,1,""],ModelFreeBasePolicy:[8,1,1,""],RandomPolicy:[8,1,1,""]},"blackbox_mpc.policies.MPCPolicy":{__init__:[8,2,1,""],act:[8,2,1,""],reset:[8,2,1,""],switch_optimizer:[8,2,1,""]},"blackbox_mpc.policies.ModelBasedBasePolicy":{__init__:[8,2,1,""],__weakref__:[8,3,1,""],act:[8,2,1,""],reset:[8,2,1,""]},"blackbox_mpc.policies.ModelFreeBasePolicy":{__init__:[8,2,1,""],__weakref__:[8,3,1,""],act:[8,2,1,""],reset:[8,2,1,""]},"blackbox_mpc.policies.RandomPolicy":{__init__:[8,2,1,""],act:[8,2,1,""],reset:[8,2,1,""]},"blackbox_mpc.trajectory_evaluators":{DeterministicTrajectoryEvaluator:[9,1,1,""],EvaluatorBase:[9,1,1,""]},"blackbox_mpc.trajectory_evaluators.DeterministicTrajectoryEvaluator":{__call__:[9,2,1,""],__init__:[9,2,1,""],evaluate_next_reward:[9,2,1,""],predict_next_state:[9,2,1,""]},"blackbox_mpc.trajectory_evaluators.EvaluatorBase":{__call__:[9,2,1,""],__init__:[9,2,1,""],evaluate_next_reward:[9,2,1,""],predict_next_state:[9,2,1,""]},"blackbox_mpc.utils":{dynamics_learning:[10,0,0,"-"],iterative_mpc:[10,0,0,"-"],pendulum:[10,0,0,"-"],recording:[10,0,0,"-"],rollouts:[10,0,0,"-"],transforms:[10,0,0,"-"]},"blackbox_mpc.utils.dynamics_learning":{learn_dynamics_from_policy:[10,4,1,""]},"blackbox_mpc.utils.iterative_mpc":{learn_dynamics_iteratively_w_mpc:[10,4,1,""]},"blackbox_mpc.utils.pendulum":{PendulumTrueModel:[10,1,1,""],pendulum_reward_function:[10,4,1,""]},"blackbox_mpc.utils.pendulum.PendulumTrueModel":{__call__:[10,2,1,""],__init__:[10,2,1,""]},"blackbox_mpc.utils.recording":{record_rollout:[10,4,1,""]},"blackbox_mpc.utils.rollouts":{perform_rollouts:[10,4,1,""]},"blackbox_mpc.utils.transforms":{default_inverse_transform_targets:[10,4,1,""],default_transform_targets:[10,4,1,""]},blackbox_mpc:{dynamics_functions:[4,0,0,"-"],dynamics_handlers:[5,0,0,"-"],environment_utils:[6,0,0,"-"],optimizers:[7,0,0,"-"],policies:[8,0,0,"-"],trajectory_evaluators:[9,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"001":[5,7,10],"00772":7,"100":0,"1000":0,"101":7,"1024":7,"128":[5,10],"150ga":7,"1604":7,"200":[0,2],"201995":7,"20particl":7,"20swarm":7,"4048":0,"454":7,"500":[0,7],"512":0,"602":7,"7989202":7,"abstract":[],"boolean":8,"case":[5,8,10],"class":[2,3,4,5,6,7,8,10],"default":[5,7,10],"float":[5,10],"function":[0,2,3,5,6,7,8,9],"import":[0,2],"int":[5,6,7,8,10],"new":0,"return":[4,5,6,7,8,9,10],"static":6,"switch":[0,8],"true":[2,3,5,8],The:[0,2,5,7,8,9,10],There:2,These:[],Useful:[2,3],Using:2,__call__:[4,7,9,10],__init__:[4,5,7,8,9,10],__weakref__:[5,6,8],_reading6:7,_sampl:10,a_par:7,a_t:[4,10],absolut:5,abund:[],achiev:7,act:[0,2,8],action:[4,5,7,8,9,10],action_sequ:9,action_spac:[0,2],action_to_execut:[0,2],actions_reward_funct:[],actions_trajectori:5,actionspac:[5,7,8,10],activation_funct:[0,4],adam:[5,10],adapt:[2,3],add_exploration_nois:7,added:[7,8,10],advantag:[],after:[0,5,7],afterward:0,agent:[2,5,6,7,8,10],aid:2,aim:[],algorithm:2,alpha:7,alpha_cov:7,analyt:[0,2],ani:[],api:2,appli:[7,9],approxim:[2,3],architectur:2,arg:8,argument:8,arnumb:7,arxiv:7,avail:2,base:[2,3],batch:[4,5,10],batch_siz:[0,5,10],batchxdim_:[8,9,10],batchxdim_u:[8,9,10],befor:[7,10],begin:[7,8],below:[0,2],best:[7,8,10],between:[7,8,10],blackbox:10,blackbox_mpc:[0,1,4,5,6,7,8,9,10],block:[0,4,7,9,10],bool:[4,5,7,8,10],bxdim_:10,calcul:[7,9],call:[4,5,7,8,9,10],can:2,capac:[],care:[0,6],cem:[0,2,7,8,10],cemoptim:[0,7],cetutori:7,challeng:[],chang:[],cheetah:0,choos:7,clone:1,cma:[0,2,8,10],cmae:7,cmaesoptim:7,code:[],collect:[0,2,3,5],com:1,comp:7,compar:0,compon:[],concaten:5,confirm:10,conjuct:2,conjunct:[],connect:4,content:[],control:[2,3,10],correspond:[9,10],cos:10,cost:[0,2],cost_func:0,coupl:[],covari:[2,3],creat:6,create_file_writ:0,cross:[2,3],current:[5,7,8,9,10],current_act:[8,9,10],current_ob:[0,2],current_policy_0:0,current_policy_:0,current_st:[5,7,8,9,10],custom:6,data:[5,8,10],def_funct:5,default_inverse_transform_target:10,default_transform_target:10,defin:[0,4,5,6,7,8,9,10],delta:[5,10],depict:[],deriv:2,design:2,despit:[],determin:7,determinist:[0,2,3],deterministic_mlp:0,deterministicdynamicsfunctionbas:5,deterministicdynamicsfunctionbaseclass:[8,10],deterministicmlp:[0,4],deterministictrajectoryevalu:[0,9],deviat:5,differ:[2,7,10],dim:[4,7,8,9,10],dim_:[4,7,9,10],dim_u:[4,8,9,10],dir:0,direct:7,directori:[5,8,10],dtheta:10,dtype:7,dynam:[2,3,7,8,9],dynamics_funct:[0,2,4,5,8,10],dynamics_handl:[0,5,8],dynamics_learn:[0,10],each:[0,4,5,7,8,9,10],eager:5,earlier:7,easiest:[0,2],edu:7,effici:[],element:5,elit:7,enabl:2,energi:7,entropi:[2,3],env:[0,2,3,10],env_action_spac:[0,2,5,7,8,10],env_class:6,env_modifi:0,env_nam:6,env_observation_spac:[0,2,5,7,8,10],environ:[0,2,3,5,7,8,10],environment_util:[0,6,10],environmentwrapp:[0,6,10],episod:[5,7,8,9,10],epoch:[0,5,10],epsilon:7,equivil:10,etc:[0,5,10],eth:[],evalu:[0,2,3,7,8],evaluate_next_reward:9,evaluatorbas:[8,9],evaluatorbaseclass:7,even:2,everyth:0,everytim:[5,10],evolutionari:[2,3],exampl:2,execut:[8,10],expand_dim:[],expected_ob:[0,2],expected_output:4,expected_reward:[0,2],explor:[8,10],exploration_nois:[0,7,8,10],extend:0,face:[],fail:[],fals:[0,5,8,10],familiar:[0,2],far:[2,8],fashion:0,featur:2,figur:[],file:10,first:[7,8,10],flexibl:2,float32:[4,5,7,8,9,10],follow:[],followup:10,forward:[],fraction:7,framework:[0,2],free:[2,3],frequenc:0,from:[0,2,5,7,8,9,10],fulli:4,func:[8,10],further:[0,10],gamma:7,gener:[],generaliz:[],get:[2,5],get_dynamics_funct:5,get_loss:4,get_validation_loss:4,gew:7,git:1,github:1,given:7,global:7,ground:4,guess:7,guid:2,gym:[0,2,5,6,7,8,10],gymenv:6,h_sigma:7,halfcheetah:2,halfcheetahenvmodifi:0,hand:[],handler:[0,2,3,8,9,10],has:[5,10],have:7,help:10,henc:[],here:0,high:[],hold:10,homework:7,horizon:[0,7,10],how:[0,5,7,8,10],http:[1,7],hw3:7,ieee:7,ieeexplor:7,implement:2,impress:[],index:[2,10],info:[0,2],inform:[2,3],initi:[7,9,10],initial_polici:[0,10],initial_velocity_fract:7,input:10,inputs_st:5,instal:2,instanc:[],instanti:[0,6],int32:[5,7,8,10],integr:[2,3],interact:[],interfac:2,intergr:[2,7],intern:9,interpret:[],invers:[5,10],inverse_transform_targets_func:5,is_norm:[0,5,10],iter:[0,5,7,8,10],iterative_mpc:[0,10],its:[7,8,9],itself:5,jhuapl:7,jsp:7,kept:7,kera:[4,5,10],known:[0,7],lab:[],lamda:7,larg:[],latest:2,layer:[0,4],learn:[2,3,5,8],learn_dynamics_from_polici:[0,10],learn_dynamics_iteratively_w_mpc:[0,10],learning_r:[5,10],length:10,level:2,like:0,limit:[],list:[5,6,8,10],load:[2,5,8,10],local:7,log:[0,2,5,8,10],log_dir:[0,5,8,10],log_path:0,lookahead:[7,10],loss:4,loss_fn:4,low:2,make:[0,2,6],make_custom_gym_env:[0,6],make_mpc_polici:[],make_standard_gym_env:[0,6],mani:[7,10],math:0,matrix:[2,3],max:7,max_iter:[0,7],maximimum:7,mdp:5,mean:7,meansquarederror:4,meant:0,method:[0,2,3],minor:[],mit:7,mlp:[0,2,3],mode:4,model:[2,3,5,7],modelbas:2,modelbasedbasepolici:[8,10],modelfreebasepolici:[8,10],modifi:0,modul:2,modular:[0,2],monitor:2,more:[2,7],mpc:[0,2,3,10],mpc_control:[],mpc_polici:[0,2,10],mpcpolici:[0,2,8],mujoco:[0,6],my_runn:[],name:[4,6,7,8,9,10],need:0,network:[2,4,5,10],neural:[5,10],new_mean:7,next:[4,5,7,8,9,10],next_observ:8,next_stat:[5,7,8,9,10],nn_optim:[5,10],nois:[7,8,10],noise_paramet:7,non:5,none:[0,4,5,8,9,10],normal:[5,8,10],note:7,num_ag:[0,2,7,8,10],num_elit:7,num_of_ag:[0,6,7,9],number:[5,6,7,8,10],number_of_ag:[0,8],number_of_initial_rollout:[0,10],number_of_refinement_step:[0,10],number_of_rollout:[0,10],number_of_rollouts_for_refin:[0,10],numpi:7,object:[4,5,6,8],observ:[5,7,8,9,10],observation_spac:[0,2],observations_trajectori:5,observationspac:[7,8,10],obtain:10,often:[5,8,10],old_mean:7,one:[7,9,10],openai:0,optim:[0,2,3,5,8,10],optimizer_arg:[8,10],optimizer_nam:[0,2,8,10],optimizer_v2:[5,10],optimizerbas:7,optimizerbaseclass:[8,10],optimz:7,option:2,org:7,ossamaahm:1,other:[2,4,8,10],out:7,output:[4,5,10],overal:[],overfit:[],own:9,packag:[0,1,2],pair:10,parallel:[2,6,7,8,10],parallel_env:0,parallelgymenv:10,paramet:[4,5,6,7,8,9,10],parellel:6,part:[],particl:[2,3],path:[2,3,10],pdf:7,pendulum:[0,2,3],pendulum_actions_reward_funct:[],pendulum_reward_funct:[0,2,10],pendulum_state_reward_funct:[],pendulumtruemodel:[0,2,10],perform:[7,10],perform_rollout:[0,10],perturb:[2,3],pi2:[2,8,10],pi2optim:7,pip:2,placehold:10,plan:[7,10],planning_horizon:[0,7,9,10],plug:0,polici:[0,2,3,10],popul:[7,9],population_s:[0,7],posit:7,possibl:[0,7],postprocess:5,power:2,predict:[2,3,4,5,7,9],predict_next_st:9,prepocess:5,preprocess:[5,9,10],previosului:0,previou:[0,5],problem:[],process:[0,5,8,10],process_input:5,process_output:5,process_state_output:5,progress:2,project:[],propens:[],prototyp:[8,9,10],provid:[2,8,10],pso:[2,8,10],psooptim:7,purpos:10,python:[4,5,10],random:[0,2,3,10],random_polici:0,random_se:6,randomli:0,randompolici:[0,8],randomsearch:[0,2,8,10],randomsearchoptim:7,rang:[0,2],rate:[5,10],raw_output:5,real:10,receiv:[5,8],record:[2,3],record_file_path:[0,10],record_rollout:[0,10],refer:[2,5,6,8],refin:[5,7,8,10],refinement_polici:[0,10],refinemnet:10,reinforc:2,rel:[5,8,10],releas:2,render:[0,2],repeat:0,repo:1,repons:5,repositori:[],repres:10,research:[0,2],reset:[0,2,7,8,10],respons:[6,7],result:[0,5],resulting_act:7,revers:10,reward:[0,2,3,5,7,8,9],reward_funct:[0,2,8,9,10],rewards_of_next_st:[7,8],rewards_trajectori:5,robot:[],rollout:[0,2,3,5],run:[0,2,4,6,7,8,10],runner:[7,8,10],s_t:[4,10],sampl:10,save:[2,5,8,10],save_model_frequ:[0,5,8,10],saved_model:0,saved_model_dir:[0,5,8,10],scalar:4,search:2,seed:6,seemingli:[],seen:[],sequenc:[7,8,9,10],set:7,set_trajectory_evalu:7,shape:[0,5,7,10],shoot:[2,3],should:[4,5,7,8,10],show:0,shown:2,sigma:7,simultan:[2,3],sin:10,single_env:[],size:[5,7,10],solut:7,some:[0,2,7],sourc:[2,4,5,6,7,8,9,10],space:[5,7,8,10],spall_stochastic_optim:7,specif:8,specifi:[6,10],split:[5,10],spsa:[2,7,8,10],spsaoptim:7,stack:[4,10],stamp:7,standard:6,start:[2,9,10],start_episod:[0,10],stat:2,state:[4,5,7,8,9,10],state_reward_funct:[],statist:[5,8,10],step:[0,2,7,8,10],still:[],stochast:[2,3],store:5,str:[0,4,8,10],strategi:[2,3],string:[5,6,7,8,9,10],structur:0,subprocvecenv:6,suggest:[],summari:[0,5,8,10],swarm:[2,3],switch_optim:[0,8],system:[0,2,5,7,8,9,10],system_dynamics_handl:[0,8,9,10],systemdynamicshandl:[0,5,8,9,10],take:[0,5,6],taken:10,tanh:0,target:[2,3,5,8],task:10,task_horizon:[0,10],tend:[],tensor:7,tensorboard:[0,2],tensorflow:[0,2,4,5,8,10],tf_func_nam:[8,9,10],tf_function:[5,8,9,10],tf_writer:[0,5,8,10],than:7,thei:[],them:[0,5],theoret:[2,3],theortic:7,theta:10,thi:[0,1,2,4,5,6,7,8,9,10],threshold:7,through:[0,2],tile:[],time:[7,8],time_step:[7,9],timestep:[7,8,9],top:[4,10],tradit:6,train:[0,2,4,5,10],train_loss:4,trainabl:5,trainer:9,traj_ac:[0,10],traj_ob:[0,10],traj_rew:[0,10],trajectori:[0,2,3,7,8],trajectory_evalu:[0,7,8,9],transform:[2,3,5],transform_targets_func:5,true_model:[0,2,5,8],truth:4,tuft:7,tutori:[0,2],tutorial_2:0,type:[4,5,6,8,9,10],underli:[],understand:[],use:[0,5,6,7,10],used:[0,2,5,6,7,8,9,10],uses:10,using:[0,2,7,8,9,10],util:[0,2,3],valid:[4,5,10],validation_loss:4,validation_split:[5,10],valu:4,vector:10,veloc:7,version:0,video:[2,3],wai:[0,2],weak:[5,6,8],web:7,weight:7,well:[5,8,10],where:[5,8,9,10],whether:4,which:[2,4,5,6,8,10],whole:4,wrap:10,wrapper:[2,3],writer:[5,8,10],written:[],www:7,www_fall_2003:7,xdim_:5,xdim_u:5,your:0,zurich:[]},titles:["Getting Started","Installation","About BlackBox_MPC","blackbox_mpc","Dynamics Functions","Dynamics Handlers","Environment Utils","Optimizers","Policies","Trajectory Evaluators","Useful Utilities"],titleterms:{"class":9,"function":[4,10],"true":[0,10],Useful:10,Using:0,about:2,adapt:7,api:0,approxim:7,base:[7,8,9,10],blackbox_mpc:[2,3],collect:10,control:[0,8],covari:7,cross:7,determinist:[4,9],dynam:[0,4,5,10],entropi:7,env:6,environ:6,evalu:9,evolutionari:7,free:8,from:1,get:0,halfcheetah:0,handler:5,indic:2,inform:7,instal:1,integr:7,latest:1,learn:[0,10],level:0,load:0,low:0,matrix:7,method:7,mlp:4,model:[0,8,10],modelbas:0,more:0,mpc:7,optim:7,particl:7,path:7,pendulum:10,perturb:7,pip:1,polici:8,predict:[0,8],random:[7,8],record:[0,10],releas:1,reward:10,rollout:10,save:0,shoot:7,simultan:7,sourc:1,start:0,stochast:7,strategi:7,swarm:7,tabl:2,target:10,theoret:7,through:1,trajectori:9,transform:10,util:[6,10],video:[0,10],wrapper:6}})