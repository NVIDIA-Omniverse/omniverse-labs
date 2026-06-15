(function () {
  const BASE = document.body.dataset.base || "";
  const DATA_VERSION =
    document.querySelector('meta[name="site-data-version"]')?.content || "";
  const DATA_URL =
    BASE +
    "data/projects.json" +
    (DATA_VERSION ? "?v=" + encodeURIComponent(DATA_VERSION) : "");

  const TYPE_LABELS = {
    article: "Article",
    project: "Project",
    sample: "Sample",
    tool: "Tool",
    concept: "Concept",
  };

  function resolveUrl(path) {
    if (!path) return null;
    if (path.startsWith("http") || path.startsWith("#")) return path;
    return BASE + path.replace(/^\//, "");
  }

  function statusClass(status) {
    const key = (status || "concept").toLowerCase();
    return "badge--status-" + key;
  }

  function renderCard(project, featured) {
    const pageUrl = resolveUrl(project.links?.page);
    const imageUrl = resolveUrl(project.image);
    const typeLabel = TYPE_LABELS[project.type] || project.type;
    const introClass = featured ? " project-card--intro" : "";

    const card = document.createElement("a");
    card.className = "project-card" + introClass;
    card.href = pageUrl || "#";
    if (project.links?.external) {
      card.target = "_blank";
      card.rel = "noopener noreferrer";
    }

    card.innerHTML =
      '<div class="project-card__media">' +
      (imageUrl
        ? '<img src="' + imageUrl + '" alt="" loading="lazy" />'
        : "") +
      "</div>" +
      '<div class="project-card__body">' +
      '<div class="project-card__meta">' +
      '<span class="badge badge--type">' +
      escapeHtml(typeLabel) +
      "</span>" +
      (project.status
        ? '<span class="badge ' +
          statusClass(project.status) +
          '">' +
          escapeHtml(project.status) +
          "</span>"
        : "") +
      "</div>" +
      '<h2 class="project-card__title">' +
      escapeHtml(project.title) +
      "</h2>" +
      (project.subtitle
        ? '<p class="project-card__subtitle">' +
          escapeHtml(project.subtitle) +
          "</p>"
        : "") +
      (project.team
        ? '<span class="project-card__team">' +
          escapeHtml(project.team) +
          "</span>"
        : "") +
      "</div>";

    if (project.accent?.length === 2) {
      card.style.setProperty(
        "--card-accent-start",
        project.accent[0]
      );
      card.style.setProperty("--card-accent-end", project.accent[1]);
      const media = card.querySelector(".project-card__media");
      if (media) {
        media.style.background =
          "linear-gradient(135deg, " +
          project.accent[0] +
          " 0%, " +
          project.accent[1] +
          " 100%)";
      }
    }

    return card;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function initHome() {
    const grid = document.getElementById("project-grid");
    const lede = document.getElementById("site-lede");
    const title = document.getElementById("site-title");
    if (!grid) return;

    fetch(DATA_URL, { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("Failed to load projects");
        return r.json();
      })
      .then(function (data) {
        if (data.site) {
          if (title && data.site.title) title.textContent = data.site.title;
          if (lede && data.site.description) lede.textContent = data.site.description;
          document.title = data.site.title + " — Omniverse Labs";
          if (data.site.dataVersion) {
            var meta = document.querySelector('meta[name="site-data-version"]');
            if (meta) meta.setAttribute("content", data.site.dataVersion);
          }
        }

        const projects = data.projects || [];
        buildFilters(projects);
        renderGrid(projects, "all");
      })
      .catch(function (err) {
        grid.innerHTML =
          '<p class="empty-state">Could not load projects. ' +
          escapeHtml(err.message) +
          "</p>";
      });
  }

  function buildFilters(projects) {
    const bar = document.getElementById("filters");
    if (!bar) return;

    const types = ["all"];
    projects.forEach(function (p) {
      if (p.type && types.indexOf(p.type) === -1) types.push(p.type);
    });

    types.forEach(function (type) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "filter-btn" + (type === "all" ? " is-active" : "");
      btn.dataset.filter = type;
      btn.textContent =
        type === "all" ? "All" : TYPE_LABELS[type] || type;
      btn.addEventListener("click", function () {
        bar.querySelectorAll(".filter-btn").forEach(function (b) {
          b.classList.toggle("is-active", b === btn);
        });
        renderGrid(projects, type);
      });
      bar.appendChild(btn);
    });
  }

  function renderGrid(projects, filter) {
    const grid = document.getElementById("project-grid");
    if (!grid) return;

    grid.innerHTML = "";

    const filtered =
      filter === "all"
        ? projects
        : projects.filter(function (p) {
            return p.type === filter;
          });

    if (filtered.length === 0) {
      grid.innerHTML =
        '<p class="empty-state">No projects match this filter yet.</p>';
      return;
    }

    const sorted = filtered.slice().sort(function (a, b) {
      if (a.featured && !b.featured) return -1;
      if (!a.featured && b.featured) return 1;
      return (b.date || "").localeCompare(a.date || "");
    });

    sorted.forEach(function (project) {
      grid.appendChild(renderCard(project, !!project.featured));
    });
  }

  if (document.getElementById("project-grid")) {
    initHome();
  }
})();
