export class Card {
  constructor() {}
}

export class SliderCard {
  constructor(elementId, data) {
    this.slider = document.getElementById(elementId);
    this.data = data;

    if (this.slider) {
      this.row = this.slider.firstElementChild;
      this.column = this.row ? this.row.firstElementChild : undefined;
      this.right = undefined;
      this.left = undefined;

      ["right", "left"].forEach(direction => {
        const [element] = this.slider.getElementsByClassName(
          `card-slider-btn-${direction}`
        );

        if (element) {
          this[direction] = element;
          this[direction].addEventListener("click", e => {
            this.scrollTo(direction, e);
          });
        }
      });

      this.activeButton();
    }
  }

  scrollTo(direction, event) {
    const { column, data } = this;

    if (column && column.childElementCount > 0) {
      const leftIncrement =
        (column.scrollWidth - column.clientWidth) /
        parseInt(data.length / 2, 10);
      let left = column.scrollLeft;

      switch (direction) {
        case "left":
          left -= leftIncrement;
          break;
        case "right":
          left += leftIncrement;
          break;
        default:
          left = 0;
          break;
      }

      column.scrollTo({ left, top: 0, behavior: "smooth" });

      setTimeout(this.activeButton, 500, this);
    }
  }

  activeButton(me) {
    const { column, left, right } = me || this;
    if (left) {
      if (column.scrollLeft > 0) {
        left.classList.remove("card-slider-hidden");
      } else {
        left.classList.add("card-slider-hidden");
      }
    }

    if (right) {
      if (column.scrollWidth - column.scrollLeft === column.clientWidth) {
        right.classList.add("card-slider-hidden");
      } else {
        right.classList.remove("card-slider-hidden");
      }
    }
  }
}
