/**
 * Scroll Indicator
 *
 * @param elementId {String}
 *
 * How to use:
 * window.onscroll = () => {
 *   onScrollIndicator("scrollIndicatorBar");
 * };
 */
export function onScrollIndicator(elementId = "scrollIndicatorBar") {
  const scrollIndicatorBar = document.getElementById(elementId);

  if (scrollIndicatorBar) {
    const winScroll =
      document.body.scrollTop || document.documentElement.scrollTop;
    const height =
      document.documentElement.scrollHeight -
      document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    scrollIndicatorBar.style.width = scrolled + "%";
  }
}
