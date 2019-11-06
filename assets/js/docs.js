import { addEventNodeClick } from "./components/tree";

import("jquery").then(async module => {
  const { default: $ } = module;

  $(document).ready(function() {
    addEventNodeClick();
  });
});
