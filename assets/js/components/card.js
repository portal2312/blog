export class Card {
  constructor() {}
}

export class SliderCard {
  constructor() {
    this.className = "card-slider";
    this.directions = ["right", "left"];

    const sliders = document.getElementsByClassName(this.className);

    for (let i = 0, len = sliders.length; i < len; i += 1) {
      const slider = sliders[i];
      slider.row = slider.firstElementChild;
      const column = slider.row.firstElementChild;
      slider.row.column = column;

      this.directions.forEach(direction => {
        const [element] = slider.getElementsByClassName(
          `${this.className}-btn-${direction}`
        );

        if (element) {
          slider[direction] = element;

          element.addEventListener("click", e => {
            this.scrollTo(element, direction, e, slider);
          });

          if (direction === "left") {
            if (column.scrollLeft === 0) {
              element.classList.add(`${this.className}-hidden`);
            } else {
              element.classList.remove(`${this.className}-hidden`);
            }
          } else if (direction === "right") {
            if (column.scrollWidth - column.scrollLeft === column.clientWidth) {
              element.classList.add(`${this.className}-hidden`);
            } else {
              element.classList.remove(`${this.className}-hidden`);
            }
          }
        }
      });
    }
  }

  scrollTo(element, direction, event, slider) {
    const { column } = slider.row;
    let left = column.scrollLeft;

    if (column && column.childElementCount) {
      switch (direction) {
        case "left":
          left -= column.offsetWidth;
          break;
        case "right":
          left += column.offsetWidth;
          break;
        default:
          left = 0;
          break;
      }
    }

    if (left <= 0) {
      slider.left.classList.add(`${this.className}-hidden`);
    } else {
      slider.left.classList.remove(`${this.className}-hidden`);
    }

    if (left + column.offsetWidth >= column.scrollWidth) {
      slider.right.classList.add(`${this.className}-hidden`);
    } else {
      slider.right.classList.remove(`${this.className}-hidden`);
    }

    column.scrollTo({ left, top: 0, behavior: "smooth" });
  }
}
