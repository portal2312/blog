// data
import codingActivityData from "../../_data/about/coding_activity.yml";
import colors from "../../_data/colors.yml";
/**
 * Sync modules:
 * 1. about.bundle.js = about.js + chart.js + chartjs-plugin-labels
 * 2. vendors~jquery.bundle.js = jquery
 */

import(
  /* webpackChunkName: "jquery" */
  "jquery"
).then(async module => {
  const { default: $ } = module;

  /**
   * Async modules:
   * 1. about.bundle.js = about.js
   * 2. vendors~jquery.bundle.js = jquery
   * 3. 0.bundle.js = chart.js
   * 4. 1.bundle.js = moment.js in the chart.js
   * 5. 2.bundle.js = chartjs-plugin-labels.js
   */
  const { default: Chart } = await import("chart.js");
  await import("chartjs-plugin-labels");

  $(document).ready(function() {
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
