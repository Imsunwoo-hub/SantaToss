import React from "react";
import { Route, Link, useHistory } from "react-router-dom";
import "./Start.css";
export const Start = () => {
  const history = useHistory();
  function goHome() {
    history.push("/Home");
  }
  return <div className="Links" onClick={goHome}></div>;
};
export default Start;
