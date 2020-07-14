import React, { useState, useEffect, Component,useRef } from "react";
import "./Home.css";
import Button from "@material-ui/core/Button";
import axios from "axios";
import MySpeechToTextComponent from "../stt/MySpeechToTextComponent";
import StopWatch from "../stopwatch/stopwatch";

//const SERVER_IP = "http://13.125.126.45:5000/image/predict/";
const SERVER_IP = "https://i02a404.p.ssafy.io/image/predict";

let bol = 0;

var SpeechRecognition =
    window.speechRecognition || window.webkitSpeechRecognition;
var recognition = new SpeechRecognition();

export const Home = () => {
  const [imgBase64, setImgBase64] = useState(""); // 파일 base64
  const [imgPath, setImgPath] = useState("/images/Logo.png");
  const [imgFile, setImgFile] = useState(null); //파일
  const [caption, setCaption] = useState(""); // 캡션
  const [b64, setB64] = useState(null);
  const [mytext, setMytext] = useState("");
  let [isRunning, setIsRunning] = useState(false);
  let [timer, setTimer] = useState(45);

  const imgdir = "https://i02a404.p.ssafy.io/images/";

  let imgArr = ["COCO_train2014_000000313360.jpg",
  "COCO_train2014_000000371839.jpg",
  "COCO_train2014_000000157202.jpg",
  "COCO_train2014_000000271999.jpg",
  "COCO_train2014_000000306822.jpg",
  "COCO_train2014_000000388527.jpg",
  "COCO_train2014_000000261381.jpg",
  "COCO_train2014_000000214232.jpg",
  "COCO_train2014_000000161963.jpg",
  "COCO_train2014_000000562330.jpg",
  "COCO_train2014_000000321508.jpg",
  "COCO_train2014_000000458908.jpg",
  "COCO_train2014_000000095081.jpg",
  "COCO_train2014_000000270431.jpg",
  "COCO_train2014_000000191283.jpg",
  "COCO_train2014_000000512948.jpg",
  "COCO_train2014_000000310203.jpg",
  "COCO_train2014_000000016950.jpg",
  "COCO_train2014_000000210136.jpg",
  "COCO_train2014_000000393854.jpg",
  "COCO_train2014_000000312486.jpg",
  "COCO_train2014_000000222913.jpg",
  "COCO_train2014_000000169330.jpg",
  "COCO_train2014_000000005355.jpg",
  "COCO_train2014_000000242167.jpg",
  "COCO_train2014_000000130637.jpg",
  "COCO_train2014_000000471966.jpg",
  "COCO_train2014_000000356810.jpg",
  "COCO_train2014_000000455287.jpg",
  "COCO_train2014_000000137029.jpg",
  "COCO_train2014_000000121965.jpg",
  "COCO_train2014_000000114424.jpg",
  "COCO_train2014_000000080041.jpg",
  "COCO_train2014_000000206530.jpg",
  "COCO_train2014_000000295267.jpg",
  "COCO_train2014_000000226383.jpg",
  "COCO_train2014_000000168405.jpg",
  "COCO_train2014_000000271298.jpg",
  "COCO_train2014_000000033656.jpg",
  "COCO_train2014_000000119294.jpg",
  "COCO_train2014_000000284886.jpg",
  "COCO_train2014_000000415203.jpg",
  "COCO_train2014_000000053022.jpg",
  "COCO_train2014_000000159562.jpg",
  "COCO_train2014_000000342887.jpg",
  "COCO_train2014_000000380724.jpg",
  "COCO_train2014_000000221717.jpg",
  "COCO_train2014_000000441488.jpg",
  "COCO_train2014_000000394104.jpg",
  "COCO_train2014_000000543803.jpg",
  "COCO_train2014_000000193166.jpg",
  "COCO_train2014_000000106023.jpg"];

 
  const randomPicture = (event) => {
    bol = 1;
    let img = new Image();
    let ran = parseInt(Math.random() * imgArr.length);
    //console.log(ran);
    setImgBase64(imgArr[ran]);
    setImgPath(imgdir+imgArr[ran])
    img.onload = function () {
      let canvas = document.createElement("canvas");
      canvas.width = this.width;
      canvas.height = this.height;
      let ctx = canvas.getContext("2d");

      ctx.drawImage(this, 0, 0);

      //let dataURL = canvas.toDataURL("image/jpeg");
      
    };
    img.src = imgdir + imgBase64;
  };

  const result = (event) => {
    let stt = document.getElementById("stt");
    stt.style.display = "none";
    recognition.stop();
    setTimer(45);
    setIsRunning(false);
    var r = document.getElementById("result");
    r.innerHTML = "";
    let result = document.getElementById("re");
    result.style.display = "block";
  };

  const home = (event) => {
    setB64(null);
    setCaption("");
    setMytext("");
    setImgFile(null);
    setImgBase64("");
    setImgPath("/images/Logo.png");
    bol = 0;
    let result = document.getElementById("re");
    result.style.display = "none";
    let main = document.getElementById("main");
    main.style.display = "block";
  };

  const startTest = async (event) => {
    var fd = new FormData();
    if(bol==1){
      console.log("1")
      fd.append("b64", null);
      fd.append("img_name", imgBase64);
    }
    else if(bol==2){
      console.log("2")
      fd.append("b64", b64);
      fd.append("img_name", null);
    }
    else {
      alert("이미지를 선택해주세요.");
      return;
    }
    //console.log(b64)
    axios({
      method: "post",
      url: `${SERVER_IP}`,
      data: fd,
    }).then((response) => {
      console.log(response.data.caption);
      setCaption(response.data.caption);
      // let content = caption;
      // content = content.replaceAll("\n", "<br/>");
      // setCaption(content);
    });
    let main = document.getElementById("main");
    main.style.display = "none";
    let stt = document.getElementById("stt");
    stt.style.display = "block";
  };

  const handleChangeFile = async (event) => {
    setImgBase64("")
    let reader = new FileReader();
    bol = 2;
    reader.onloadend = () => {
      const base64 = reader.result;
      setB64(reader.result);
      if (base64) {
        setImgBase64(base64.toString());
        setImgPath(base64.toString());
      }
    };
    if (event.target.files[0]) {
      //console.log(event.target.files[0]);
      reader.readAsDataURL(event.target.files[0]);
      setImgFile(event.target.files[0]);
    }
  };

  useInterval(() => {
    if (isRunning) {
      setTimer(timer - 1);
      if(timer == 1){
        setIsRunning(!isRunning);
        recognition.stop();
      }
    }
  }, 1000);

  //Speech To Text Method Start
  
  function startConverting() {
    setIsRunning(!isRunning);
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.start();
    var finalTranscripts = "";
    var r = document.getElementById("result");

    recognition.onresult = function (event) {
  
      var interimTranscripts = "";
      for (var i = event.resultIndex; i < event.results.length; i++) {
        var transcript = event.results[i][0].transcript;
        transcript.replace("\n", "<br>");
        if (event.results[i].isFinal) {
          finalTranscripts += transcript;
        } else {
          interimTranscripts += transcript;
        }
      }
      r.innerHTML =
        finalTranscripts +
        '<span style="color:#999">' +
        interimTranscripts +
        "</span>";
      setMytext(finalTranscripts);
 //
    };
  };
  //Speech To Text method END
  function stopConverting() {
    setIsRunning(false);
    recognition.stop();
  };

  return (
    <div className="contents">
      <h1>산타 토스 : PART 2</h1>
      <div
        className="imgbox"
        style={{
          backgroundColor: "#efefef",
        }}
      >
        <img src={imgPath} style={{}} />
      </div>
      <div id="main">
        <div>
          <br />
          <label className="inputLabel">
            사진 고르기
            <input
              type="file"
              name="imgFile"
              id="imgFile"
              onChange={handleChangeFile}
            />
          </label>
          &nbsp;
          <Button
            className="btns"
            style={{
              backgroundColor: "#44679f",
              color: "white",
              width: "101.19px",
              height: "39.2px",
              boxShadow: "none",
              fontFamily: "Noto Sans KR, sans-serif",
              fontSize: "16px",
            }}
            onClick={randomPicture}
          >
            랜덤 사진
          </Button>
        </div>
        <div className="startTest">
          <Button
            style={{
              backgroundColor: "#e03131",
              color: "white",
              width: "100%",
              height: "39.2px",
              boxShadow: "none",
              fontFamily: "Noto Sans KR, sans-serif",
              fontSize: "16px",
            }}
            onClick={startTest}
          >
            테스트 시작
          </Button>
        </div>
      </div>
      <div id="stt" style={{ display: "none" }}>
        <h3 align="center">내 발음은 이렇게 들립니다</h3>
        <div id="result" className="result"></div>
        <div className="stopwatch-time">{timer}</div>
        <div className="Record-btns">
          <Button
            style={{
              backgroundColor: "#44679f",
              color: "white",
              width: "80px",
              height: "auto",
              boxShadow: "none",
              fontFamily: "Noto Sans KR, sans-serif",
              fontSize: "16px",
            }}
            onClick={startConverting}
          >
            시작
          </Button>
          &nbsp; &nbsp; &nbsp; &nbsp;
          <Button
            style={{
              backgroundColor: "#44679f",
              color: "white",
              width: "80px",
              height: "auto",
              boxShadow: "none",
              fontFamily: "Noto Sans KR, sans-serif",
              fontSize: "16px",
            }}
            onClick={stopConverting}
          >
            중지
          </Button>
        </div>
        <div className="result">
        <Button
          style={{
            marginTop: "5px",
            backgroundColor: "#e03131",
            color: "white",
            width: "350px",
            height: "39.2px",
            boxShadow: "none",
            fontFamily: "Noto Sans KR, sans-serif",
            fontSize: "16px",
          }}
          onClick={result}
        >
          결과 확인
        </Button>
        </div>
      </div>

      <div id="re" style={{display:"none", marginTop:"5px" }}>
      <div className="replay">
      <div style={{textAlign:"left", width:"100%" }}>예상 답변</div>
      
      <div style={{textAlign:"left", width:"100%", backgroundColor:"#f1f3f5" }}>
        {
          caption.split('\n').map( line => {
            return (<span>{line}.<br/><br/></span>)
          })
        }
        </div>
      
      <div style={{textAlign:"left", width:"100%", marginTop:"3px"}}> 내 문장 </div> 
      <div style={{textAlign:"left", width:"100%", backgroundColor:"#f1f3f5"}}>{mytext}.
      </div>
    
      <Button
         style={{
          backgroundColor: "#e03131",
          color: "white",
          width: "350px",
          height: "39.2px",
          boxShadow: "none",
          fontFamily: "Noto Sans KR, sans-serif",
          fontSize: "16px",
          marginTop:"3px"
        }}
          onClick={home}>
          다시 하기
        </Button>
        </div>
      </div>

    </div>
  );
};


function useInterval(callback, delay) {
  const savedCallback = useRef();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    function tick() {
      savedCallback.current();
    }
    if (delay !== null) {
      let id = setInterval(tick, delay);
      return () => clearInterval(id);
    }
  }, [delay]);
}

export default Home;
