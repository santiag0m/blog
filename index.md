---
layout: default
---

# Welcome

## Latest Posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | relative_url}}">{{ post.title }}</a>
      {{ post.excerpt}}
    </li>
  {% endfor %}
</ul>
