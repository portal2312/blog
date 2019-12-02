import { SliderCard } from "./components/card";

import("jquery").then(async module => {
  const { default: $ } = module;

  $(document).ready(function() {
    new SliderCard();
  });
});
