import { SliderCard } from "./components/card";
import postsData from "../../_data/posts.json";

import("jquery").then(async module => {
  const { default: $ } = module;

  $(document).ready(function() {
    Object.keys(postsData).forEach(category => {
      const data = postsData[category];
      new SliderCard(`card-slider-${category}`, data);
    });
  });
});
