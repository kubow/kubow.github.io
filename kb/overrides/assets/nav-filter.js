// Filter box on top of the left navigation tree.
// Hides nav entries that don't match and expands sections containing matches.
(function () {
  var STORAGE_KEY = "nav-filter";

  function ownLabel(li) {
    for (var el = li.firstElementChild; el; el = el.nextElementSibling) {
      if (el.matches(".md-nav__link, .md-nav__container")) return el;
    }
    return null;
  }

  function childItems(li) {
    var nav = li.querySelector(":scope > nav");
    return nav ? nav.querySelectorAll(":scope > ul > li.md-nav__item") : [];
  }

  function haystack(li) {
    var label = ownLabel(li);
    if (!label) return "";
    var link = label.matches("a") ? label : label.querySelector("a");
    var href = link ? decodeURIComponent(link.getAttribute("href") || "") : "";
    return (label.textContent + " " + href).toLowerCase();
  }

  // Returns true when the item (or a descendant) matches every term.
  function apply(li, terms, forceShow) {
    var text = haystack(li);
    var selfMatch = terms.every(function (t) { return text.indexOf(t) !== -1; });
    var anyChild = false;
    childItems(li).forEach(function (child) {
      if (apply(child, terms, forceShow || selfMatch)) anyChild = true;
    });
    var visible = forceShow || selfMatch || anyChild;
    li.classList.toggle("nav-filter--hidden", !visible);
    var toggle = li.querySelector(":scope > input.md-nav__toggle");
    if (toggle && anyChild) toggle.checked = true;
    return selfMatch || anyChild;
  }

  function run(nav, query, empty) {
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    var top = nav.querySelectorAll(":scope > ul > li.md-nav__item");
    if (!terms.length) {
      top.forEach(function (li) { reset(li); });
      empty.hidden = true;
      return;
    }
    var found = false;
    top.forEach(function (li) { if (apply(li, terms, false)) found = true; });
    empty.hidden = found;
  }

  function reset(li) {
    li.classList.remove("nav-filter--hidden");
    var toggle = li.querySelector(":scope > input.md-nav__toggle");
    if (toggle && toggle.dataset.navFilterInitial !== undefined) {
      toggle.checked = toggle.dataset.navFilterInitial === "1";
    }
    childItems(li).forEach(reset);
  }

  function init() {
    var nav = document.querySelector(".md-sidebar--primary .md-nav--primary");
    if (!nav || nav.querySelector(".nav-filter")) return;

    nav.querySelectorAll("input.md-nav__toggle").forEach(function (t) {
      t.dataset.navFilterInitial = t.checked ? "1" : "0";
    });

    var box = document.createElement("div");
    box.className = "nav-filter";
    box.innerHTML =
      '<input type="search" class="nav-filter__input" placeholder="Filter pages…" aria-label="Filter navigation">' +
      '<div class="nav-filter__empty" hidden>No matching pages</div>';
    var list = nav.querySelector(":scope > ul");
    nav.insertBefore(box, list);
    // Stick right below the (also sticky) sidebar title.
    var title = nav.querySelector(":scope > .md-nav__title");
    if (title) box.style.top = title.offsetHeight + "px";

    var input = box.querySelector("input");
    var empty = box.querySelector(".nav-filter__empty");
    input.addEventListener("input", function () {
      try { sessionStorage.setItem(STORAGE_KEY, input.value); } catch (e) {}
      run(nav, input.value, empty);
    });
    // Keep Material's global keyboard shortcuts from stealing keystrokes.
    input.addEventListener("keydown", function (e) { e.stopPropagation(); });

    var saved = "";
    try { saved = sessionStorage.getItem(STORAGE_KEY) || ""; } catch (e) {}
    if (saved) {
      input.value = saved;
      run(nav, saved, empty);
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
