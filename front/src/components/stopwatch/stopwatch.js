import React, { useState, useEffect, useRef } from "react";
import Button from "@material-ui/core/Button";

const Stopwatch = (props) => {
  let [isRunning, setIsRunning] = useState(false);
  let [timer, setTimer] = useState(45);

  useInterval(() => {
    if (isRunning) {
      setTimer(timer - 1);
    }
  }, 1000);

  return (
    <div className="stopwatch">
      <div className="stopwatch-time">{timer}</div>
      <Button onClick={() => setIsRunning(!isRunning)}>
        {isRunning ? "중지" : "시작"}
      </Button>
      <Button onClick={() => setTimer(45)}>Reset</Button>
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

export default Stopwatch;