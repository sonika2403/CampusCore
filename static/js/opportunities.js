document.addEventListener("DOMContentLoaded", function () {
  loadPosts();

  document.getElementById("postForm").addEventListener("submit", function (e) {
    e.preventDefault();
    submitPost();
  });
});

// Keep these existing functions
function loadPosts() {
  fetch("/posts")
    .then((response) => response.json())
    .then((posts) => {
      const postsContainer = document.getElementById("posts");
      postsContainer.innerHTML = "";
      posts.reverse().forEach((post) => {
        postsContainer.appendChild(createPostElement(post));
      });
    });
}

function createPostElement(post) {
  const postDiv = document.createElement("div");
  postDiv.className = "post";
  postDiv.innerHTML = `
      <div class="post-header">
          <h3 class="post-title">${post.title}</h3>
          <button class="like-button" onclick="likePost('${post.timestamp}')">
              ❤ ${post.likes}
          </button>
      </div>
      <div class="post-meta">
          <p>By: ${post.name} | ${post.email}</p>
          <p>Posted on: ${post.timestamp}</p>
      </div>
      <p class="post-description">${post.description}</p>
      ${
        post.form_link
          ? `<a href="${post.form_link}" target="_blank" class="form-link">Open Google Form</a>`
          : ""
      }
  `;
  return postDiv;
}

function submitPost() {
  const formData = {
    name: document.getElementById("name").value,
    email: document.getElementById("email").value,
    title: document.getElementById("title").value,
    description: document.getElementById("description").value,
    form_link: document.getElementById("form_link").value,
  };

  fetch("/add_post", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("postForm").reset();
      loadPosts();
    });
}

function likePost(timestamp) {
  fetch("/like_post", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ timestamp: timestamp }),
  })
    .then((response) => response.json())
    .then((data) => {
      loadPosts();
    });
}
