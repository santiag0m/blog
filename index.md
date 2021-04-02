# Welcome

## Latest Posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | relative_url}}">{{ post.title }}</a>
      {{ post.content | markdownify | strip_html | truncatewords: 50 }}
    </li>
  {% endfor %}
</ul>
