import React from "react";
import Button from "@material-ui/core/Button";
import { useState, useEffect, useRef } from "react";


export const MySpeechToTextComponent = () => {

  let [isRunning, setIsRunning] = useState(false);
  let [timer, setTimer] = useState(45);
  let [text, setText] = useState("");

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
  var SpeechRecognition =
    window.speechRecognition || window.webkitSpeechRecognition;
  var recognition = new SpeechRecognition();
  function startConverting() {
    setIsRunning(!isRunning);
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.start();
    console.log("Y");
    var finalTranscripts = "";
    var r = document.getElementById("result");

    recognition.onresult = function (event) {
      console.log("YYY");
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
      setText(finalTranscripts);
      console.log(r); //
    };
  }
  //Speech To Text method END
  function stopConverting() {
    setIsRunning(!isRunning)
    recognition.stop();
  }
  return (
    <div>
      <br />
      <h3 align="center">내 발음은 이렇게 들립니다</h3>
      <div id="result" className="result"></div>
      <br />
      <div className="stopwatch-time">{timer}</div>
      <div className="Record-btns">
        <Button
          style={{
            backgroundColor: "#3c3c3c",
            color: "white",
            width: "auto",
            height: "auto",
            boxShadow: "none",
          }}
          onClick={startConverting}
        >
        시작
        </Button>
        &nbsp; &nbsp; &nbsp;
        <Button
          style={{
            backgroundColor: "#3c3c3c",
            color: "white",
            width: "auto",
            height: "auto",
            boxShadow: "none",
          }}
          onClick={stopConverting}
        >
          중지
        </Button>
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

export default MySpeechToTextComponent;
