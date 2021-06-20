// ml5.js: Pose Classification
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/Courses/ml5-beginners-guide/7.2-pose-classification.html
// https://youtu.be/FYgYyq-xqAw

// All code: https://editor.p5js.org/codingtrain/sketches/JoZl-QRPK

// Separated into three sketches
// 1: Data Collection: https://editor.p5js.org/codingtrain/sketches/kTM0Gm-1q
// 2: Model Training: https://editor.p5js.org/codingtrain/sketches/-Ywq20rM9
// 3: Model Deployment: https://editor.p5js.org/codingtrain/sketches/c5sDNr8eM


let video;
let stdvideo;
let poseNet;
let stdposeNet;
let pose;
let stdpose;
let skeleton;
let textX = 720;
let brain;
let poseLabel = "A";
let outputs=[];
let who;
let acc;

function find_angle(A_x,A_y,B_x,B_y,C_x,C_y) {
    var AB = Math.sqrt(Math.pow(B_x-A_x,2)+ Math.pow(B_y-A_y,2));    
    var BC = Math.sqrt(Math.pow(B_x-C_x,2)+ Math.pow(B_y-C_y,2)); 
    var AC = Math.sqrt(Math.pow(C_x-A_x,2)+ Math.pow(C_y-A_y,2));
    return (Math.acos((BC*BC+AB*AB-AC*AC)/(2*BC*AB))*180)/Math.PI;
}
  
function setup() {
  createCanvas(640, 480);
  //stdvideo = createVideo('ronaldo.mp4');
  video = createVideo('testt.mp4');
  //console.log(video);
  video.loop();
  video.hide();
  //stdvideo.noloop();
  //stdvideo.hide();
 // video = createCapture(VIDEO);
 // video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  //stdposeNet = ml5.poseNet(stdvideo, modelLoaded);
  //stdposeNet.on('stdpose', gotstdPoses);
  //console.log('?');

  let options = {
    inputs: 12,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  const modelInfo = {
    model: 'modelmk2/modelmk2.json',
    metadata: 'modelmk2/modelmk2_meta.json',
    weights: 'modelmk2/modelmk2.weights.bin',
  };
  brain.load(modelInfo, brainLoaded);
}



function brainLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    /*for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }*/
    let angleinputs = new Array(12);
      
      angleinputs[0]=find_angle(pose.leftElbow.x,pose.leftElbow.y,pose.leftShoulder.x,pose.leftShoulder.y,pose.leftHip.x,pose.leftHip.y);
      angleinputs[1]=find_angle(pose.rightElbow.x,pose.rightElbow.y,pose.rightShoulder.x,pose.rightShoulder.y,pose.rightHip.x,pose.rightHip.y);
      angleinputs[2]=find_angle(pose.leftHip.x,pose.leftHip.y,pose.leftShoulder.x,pose.leftShoulder.y,pose.rightShoulder.x,pose.rightShoulder.y);
      angleinputs[3]=find_angle(pose.rightHip.x,pose.rightHip.y,pose.rightShoulder.x,pose.rightShoulder.y,pose.leftShoulder.x,pose.leftShoulder.y);
      angleinputs[4]=find_angle(pose.leftWrist.x,pose.leftWrist.y,pose.leftElbow.x,pose.leftElbow.y,pose.leftShoulder.x,pose.leftShoulder.y);
      angleinputs[5]=find_angle(pose.rightWrist.x,pose.rightWrist.y,pose.rightElbow.x,pose.rightElbow.y,pose.rightShoulder.x,pose.rightShoulder.y);
      angleinputs[6]=find_angle(pose.leftShoulder.x,pose.leftShoulder.y,pose.leftHip.x,pose.leftHip.y,pose.rightHip.x,pose.rightHip.y);
      angleinputs[7]=find_angle(pose.rightShoulder.x,pose.rightShoulder.y,pose.rightHip.x,pose.rightHip.y,pose.leftHip.x,pose.leftHip.y);
      angleinputs[8]=find_angle(pose.leftKnee.x,pose.leftKnee.y,pose.leftHip.x,pose.leftHip.y,pose.rightHip.x,pose.rightHip.y);
      angleinputs[9]=find_angle(pose.rightKnee.x,pose.rightKnee.y,pose.rightHip.x,pose.rightHip.y,pose.leftHip.x,pose.leftHip.y);
      angleinputs[10]=find_angle(pose.leftAnkle.x,pose.leftAnkle.y,pose.leftKnee.x,pose.leftKnee.y,pose.leftHip.x,pose.leftHip.y);
      angleinputs[11]=find_angle(pose.rightAnkle.x,pose.rightAnkle.y,pose.rightKnee.x,pose.rightKnee.y,pose.rightHip.x,pose.rightHip.y);
      for (let i=0;i<angleinputs.length;i++) {
        inputs.push(angleinputs[i]);
      }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}


function gotResult(error, results) {

  let count = 0;
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label.toUpperCase();
    if(outputs.length<100){
      outputs.push(poseLabel);
      count++;
    }
    //text(poseLabel);
  }
  //console.log(results[0].confidence);
  classifyPose();
  
}

function gotOutputs() {
  console.log(outputs.length);
  let mf = 1;
  let m = 0;
  let item;
  for (let i=0; i<outputs.length; i++)
  {
        for (let j=i; j<outputs.length; j++)
        {
                if (outputs[i] == outputs[j])
                 m++;
                if (mf<m)
                {
                  mf=m; 
                  item = outputs[i];
                }
        }
        m=0;
  }
  console.log(outputs.length)
  acc=mf;
  return item;
}

function mousePressed(){
  noLoop();
}
function mouseReleased(){
  loop();
}
function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
 // console.log(poses);
}


function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  push();
  //translate(video.width,0);

  //scale(-1,1);
  image(video, 0, 0,video.width,video.height);
  if(outputs.length<100)
    gotOutputs();
  else{
    who=gotOutputs();
    console.log(who);
    console.log(acc+'%');
  }
   
  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 4, 4);
    }
  }

  
  pop();
  
  fill(255, 255, 255);
  noStroke();
  textSize(32);
  textAlign(CENTER, CENTER);
  //text(poseLabel, width / 2 , 30);
  if(who){
    text(who, width / 2, 30);
  }
  else{
    text("Classifying...",width/2,30);
  }
  //
  //console.log(who+'like');
}

