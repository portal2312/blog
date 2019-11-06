/**
 * Tree
 * Nodes in tree add click event.
 * @param className {String}
 *
 * How to use:
 * $(document).ready(function() {
 *   addEventNodeClick();
 * });
 */
export function addEventNodeClick(className = "tree") {
  const tree = document.getElementsByClassName(className);

  if (tree) {
    const nodes = tree[0].getElementsByClassName("node");

    for (let i = 0, len = nodes.length; i < len; i += 1) {
      nodes[i].addEventListener("click", function() {
        this.parentElement.querySelector(".nested").classList.toggle("active");
      });
    }
  }
}
