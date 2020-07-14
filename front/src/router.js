import * as React from "react";
import { Route, Switch } from "react-router-dom";
import Home from "./components/home/Home";
import Start from "./components/start/Start";

class MyRouter extends React.Component {
  render() {
    return (
      <Switch>
        {/* <Route exact path="/" component={Start} /> */}
        <Route exact path="/" component={Home} />
      </Switch>
    );
  }
}
export default MyRouter;
