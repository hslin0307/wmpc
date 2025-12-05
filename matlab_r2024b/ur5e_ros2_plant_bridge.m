%% ur5e_ros2_plant_bridge.m  (no HistoryDepth; callback TRAJ; R2024b OK)
clear; clc;

%% 0) User Config ----------------------------------------------------------
TOPIC_JOINT_TRAJ = "/scaled_joint_trajectory_controller/joint_trajectory"; % trajectory_msgs/JointTrajectory
TOPIC_VEL_CMD    = "/forward_velocity_controller/commands";                % std_msgs/Float64MultiArray (6)
TOPIC_FORCE_CMD  = "/force_mode_controller/commands";                      % geometry_msgs/Wrench or WrenchStamped
TOPIC_JOINT_STS  = "/joint_states";                                        % sensor_msgs/JointState

URDF = fullfile(getenv('HOME'),'ur5e.urdf');
BASE = "base_link"; 
TCP  = "tool0";

dt   = 0.004;              % 250 Hz
simDuration = inf;
viz_period  = 0.04;

% Joint PD
Kp = diag([120 120 90 50 30 10]);
Kd = diag([12  12  8  5  3  1]);

% Simulation clamps
vlim = [3.14 3.14 3.14 3.2 3.2 3.2];   % rad/s
elim = [150 150 150 28 28 28];         % Nm

% TWIST protection
VLIN_MAX = 0.25;   % m/s
VANG_MAX = 1.5;    % rad/s
WS_X = [0.05  0.80];
WS_Y = [-0.60 0.60];
WS_Z = [0.05  0.85];
Kwall = 400; Bwall = 20;
Kns = 0.0;

%% 1) Load Robot -----------------------------------------------------------
robot = importrobot(URDF,'DataFormat','row');
robot.Gravity = [0 0 -9.81];

% movable joint names only
names = {};
for i = 1*numel(robot.Bodies)
end
for i = 1:numel(robot.Bodies)
    j = robot.Bodies{i}.Joint;
    if ~strcmpi(j.Type,'fixed'), names{end+1} = char(j.Name); end %#ok<AGROW>
end
if numel(names) ~= 6
    error("Expected 6 movable joints, got %d. Names: %s", numel(names), strjoin(names,", "));
end
n = numel(names);

q  = zeros(1,n); 
qd = zeros(1,n);

figure('Name','UR5e Plant'); 
show(robot,q,'Frames','off','PreservePlot',false); title('UR5e Plant'); drawnow;

%% 2) ROS 2 Setup ----------------------------------------------------------
node = ros2node("matlab_ur5e_plant");

pubJS = ros2publisher(node, TOPIC_JOINT_STS,'sensor_msgs/JointState');
js = ros2message('sensor_msgs/JointState'); 
js.name = names;

% JointTrajectory with callback (no NV-args)
traj_inbox = struct('T',[],'Q',[],'t0',0);
traj_flag  = false;
subTraj = ros2subscriber(node, TOPIC_JOINT_TRAJ, 'trajectory_msgs/JointTrajectory', ...
                         @(msg) trajCallback(msg));

% Velocity / Force basic subscribers (no NV-args)
subWrench  = trySub(node, TOPIC_FORCE_CMD, 'geometry_msgs/Wrench');
subWrenchS = trySub(node, TOPIC_FORCE_CMD, 'geometry_msgs/WrenchStamped');

% --- Velocity subscribers: support 3 types ---
subVelArr = ros2subscriber(node, TOPIC_VEL_CMD, 'std_msgs/Float64MultiArray');  % [vx vy vz wx wy wz]
subVelTw  = [];
subVelTws = [];
try,  subVelTw  = ros2subscriber(node, TOPIC_VEL_CMD, 'geometry_msgs/Twist');        end
try,  subVelTws = ros2subscriber(node, TOPIC_VEL_CMD, 'geometry_msgs/TwistStamped'); end


%% 3) State / Modes --------------------------------------------------------
mode   = "HOLD";        % IDLE | HOLD | TRAJ | TWIST | WRENCH
q_hold = q;
lastF  = zeros(1,6);
lastV  = zeros(1,6);
Tdes_accum = getTransform(robot, q, char(TCP), char(BASE));

tic; next = 0; last_viz = 0;

%%% LOGGING: 初始化軌跡 log（只存少量資料）
trajLog.t     = [];         % 時間
trajLog.q     = [];         % 關節角 (1x6)
trajLog.mode  = strings(0,1); % 模式: "TRAJ"/"TWIST"/"WRENCH"
trajLog.v_cmd = [];         % 最近一次 base 速度指令 (1x6)
trajLog.f_cmd = [];         % 最近一次 base 力指令 (1x6)

%% 4) Main Loop ------------------------------------------------------------
lastVelTime = -inf;
velTimeout  = 1.0;   % 超過 0.1 s 沒新 velocity 就停

while toc < simDuration
    now = toc;
    if now < next, pause(0.0005); continue; end
    next = next + dt;

    % (A) JointTrajectory inbox from callback
    if traj_flag
        traj = traj_inbox;  traj_flag = false;  mode = "TRAJ";
    end

    % (B) Velocity (Float64MultiArray / Twist / TwistStamped)
    velGot = false; v6 = [];
    msg = tryRecv(subVelArr, 0.05);
    if ~isempty(msg)
        data = double(msg.data(:)).';
        if numel(data) >= 6
            v6 = data(1:6); velGot = true;
        end
    end
    if ~velGot && ~isempty(subVelTw)
        msg = tryRecv(subVelTw, 0.05);
        if ~isempty(msg)
            v6 = [msg.linear.x msg.linear.y msg.linear.z ...
                  msg.angular.x msg.angular.y msg.angular.z];
            v6 = double(v6); velGot = true;
        end
    end
    if ~velGot && ~isempty(subVelTws)
        msg = tryRecv(subVelTws, 0.05);
        if ~isempty(msg)
            v6 = [msg.twist.linear.x msg.twist.linear.y msg.twist.linear.z ...
                  msg.twist.angular.x msg.twist.angular.y msg.twist.angular.z];
            v6 = double(v6); velGot = true;
        end
    end

    if velGot
        lastV = v6;
        mode = "TWIST";
        lastVelTime = now;

    elseif mode == "TWIST" && (now - lastVelTime) > velTimeout
        % === 這裡是關鍵 ===
        % Twist 結束 → 切回 HOLD，同時「鎖定當前 q 做為新的 hold 姿勢」
        mode   = "HOLD";
        lastV  = zeros(1,6);
        q_hold = q;         % ★ 不再拉回初始 0，而是 hold 現在姿勢
        qd     = zeros(1,n);   % ★ 強制把速度歸零，HOLD 後不再滑行
        fprintf("[MODE] TWIST -> HOLD at t=%.3f s\n", now);
    end


    % (C) Wrench / WrenchStamped
    msgW  = tryRecv(subWrench, 0.0);
    msgWS = tryRecv(subWrenchS,0.0);
    if ~isempty(msgW)
        lastF = [msgW.force.x msgW.force.y msgW.force.z msgW.torque.x msgW.torque.y msgW.torque.z];
        mode  = "WRENCH";
    elseif ~isempty(msgWS)
        lastF = [msgWS.wrench.force.x msgWS.wrench.force.y msgWS.wrench.force.z ...
                 msgWS.wrench.torque.x msgWS.wrench.torque.y msgWS.wrench.torque.z];
        mode  = "WRENCH";
    end

    % (D) Control torque
    tau = zeros(1,n);
    switch mode
        case "TRAJ"
            trel = now - traj.t0;
            if trel >= traj.T(end)
                q_ref  = traj.Q(end,:);
                qd_ref = zeros(1,n);
            else
                idx = find(traj.T >= trel, 1);
                if idx==1
                    q_ref = traj.Q(1,:);
                else
                    t0 = traj.T(idx-1); t1=traj.T(idx);
                    alpha = (trel - t0)/(t1 - t0 + eps);
                    q_ref = (1-alpha)*traj.Q(idx-1,:) + alpha*traj.Q(idx,:);
                end
                qd_ref = zeros(1,n);
            end
            e  = (q_ref - q).'; ed = (qd_ref - qd).';
            tau = (Kp*e + Kd*ed).';


        case "TWIST"
            % lastV: 期望 TCP twist (在 BASE frame)
            v = lastV(:);                   % 6x1
            v(1:3) = clamp_vec(v(1:3), VLIN_MAX);
            v(4:6) = clamp_vec(v(4:6), VANG_MAX);
    
            % 幾何 Jacobian：BASE frame，TCP body
            J = geometricJacobian(robot, q, char(TCP));   % 6x6
    
            % Damped least-squares 反解: qd_des = J^+ v
            lambda  = 1e-3;                               % 小阻尼，避開奇異
            A = J*J.' + (lambda^2)*eye(6);                % 6x6
            qd_des = J.' * (A \ v);                       % 6x1
    
            % 只做「速度誤差 PD」，位置不再丟 IK
            e_v  = (qd_des.' - qd).';                     % n×1
            tau  = (Kd * e_v).';                          % 1×n

        case "WRENCH"
            J = geometricJacobian(robot, q, char(TCP));
            tau = (J.' * lastF(:)).';

        case "HOLD"
            e  = (q_hold - q).'; ed = (zeros(1,n) - qd).';
            tau_pd = (Kp*e + Kd*ed).';
            C = velocityProduct(robot, q, qd);
            G = gravityTorque(robot, q);
            tau = (C + G + tau_pd);

        otherwise
            tau = zeros(1,n);
    end

    % (E) Dynamics & integrate
    tau = min(max(tau, -elim), elim);
    M = massMatrix(robot, q);
    C = velocityProduct(robot, q);
    G = gravityTorque(robot, q);
    qdd = (M \ (tau(:) - C(:) - G(:))).';

    qd = qd + qdd*dt;  qd = min(max(qd, -vlim), vlim);
    q  = q  + qd*dt;

    %%% LOGGING: 只在非 HOLD 狀態紀錄一筆
    if mode ~= "HOLD"
        trajLog.t(end+1,1)     = now;          %#ok<AGROW>
        trajLog.q(end+1,1:n)   = q;            %#ok<AGROW>
        trajLog.mode(end+1,1)  = mode;         %#ok<AGROW>
        trajLog.v_cmd(end+1,1:6) = lastV;      %#ok<AGROW>
        trajLog.f_cmd(end+1,1:6) = lastF;      %#ok<AGROW>
    end

    % (F) Publish joint_states
    js.header.stamp.sec     = int32(floor(now));
    js.header.stamp.nanosec = uint32((now - floor(now))*1e9);
    js.position = q; js.velocity = qd;
    send(pubJS, js);

    % (G) Visualize
    if now - last_viz >= viz_period
        show(robot, q, 'Frames','off','PreservePlot',false);
        title(sprintf("%s  t=%.2f s", mode, now)); drawnow;
        last_viz = now;
    end
end

%% 5) Callbacks / Helpers --------------------------------------------------
function trajCallback(msg)
    persistent plantOrder
    persistent traj_inbox traj_flag
    if isempty(plantOrder)
        js = evalin('base','js');
        plantOrder = string(js.name);
    end
    srcOrder = string(msg.joint_names);
    pts = msg.points;
    if isempty(pts), return; end

    idx = zeros(1, numel(plantOrder));
    for ii = 1:numel(plantOrder)
        k = find(srcOrder == plantOrder(ii), 1);
        if isempty(k), k = 1; end
        idx(ii) = k;
    end
    Tseg = arrayfun(@(p) double(p.time_from_start.sec)+double(p.time_from_start.nanosec)*1e-9, pts);
    Qseg = zeros(numel(pts), numel(plantOrder));
    for r = 1:numel(pts)
        pos = double(pts(r).positions(:)).';
        Qseg(r,:) = pos(idx);
    end
    traj_inbox.T  = Tseg(:).';
    traj_inbox.Q  = Qseg;
    traj_inbox.t0 = toc;    
    assignin('base','traj_inbox',traj_inbox);
    assignin('base','traj_flag',true);
    fprintf("[TRAJ RX] points=%d, names=%s, T_end=%.3fs\n", numel(pts), strjoin(cellstr(srcOrder),","), Tseg(end));
end

function sub = trySub(node, topic, type)
    try, sub = ros2subscriber(node, topic, type);
    catch, sub = []; end
end

function msg = tryRecv(sub, timeout)
    if isempty(sub), msg = []; return; end
    try, msg = receive(sub, timeout); catch, msg = []; end
end

function vout = clamp_vec(vin, vmax)
    n = norm(vin);
    if n <= vmax || n < 1e-12, vout = vin; else, vout = vin * (vmax / n); end
end

function pnext = soft_wall(p, rng, K, B, dt)
    pmin = rng(1); pmax = rng(2);
    persistent vp; if isempty(vp), vp = 0; end
    vp = (1-0.5)*vp;
    f = 0;
    if p < pmin, f = K*(pmin - p) - B*vp;
    elseif p > pmax, f = -K*(p - pmax) - B*vp;
    end
    vp = vp + f*dt;
    pnext = p + f*dt;
    pnext = min(max(pnext, pmin), pmax);
end
