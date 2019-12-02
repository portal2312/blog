import { addEventNodeClick } from "./components/tree";
import codingActivityData from "../../_data/about/coding_activity.yml";
import colors from "../../_data/colors.yml";

import("jquery").then(async module => {
  const { default: Chart } = await import("chart.js");
  await import("chartjs-plugin-labels");

  const { default: $ } = module;
  $(document).ready(function() {
    addEventNodeClick();

    const chartEl = document.getElementById("coding-activity-chart");
    if (chartEl) {
      const labels = [];
      const data = [];
      codingActivityData.items.forEach(item => {
        labels.push(item.label);
        data.push(item.value);
      });
      const backgroundColor = Object.values(colors).splice(0, data.length + 1);

      new Chart(chartEl, {
        type: "doughnut",
        data: {
          labels: labels,
          datasets: [
            {
              label: "",
              data: data,
              backgroundColor: backgroundColor
            }
          ]
        },
        options: {
          plugins: {
            labels: {
              fontColor: "#ffffff",
              fontSize: 14,
              fontStyle: "bold",
              render: kwargs =>
                [
                  `${kwargs.label}:`,
                  `${kwargs.value} (${kwargs.percentage}%)`
                ].join("\n")
            }
          }
        }
      });
    }
  });
});
