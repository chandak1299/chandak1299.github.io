---
layout: page
title: Publications
permalink: /publications/
---
{% comment %}
  Papers, the thesis and talks come from _data/publications.yml,
  _data/thesis.yml and _data/talks.yml;
  edit those files to add or change entries.
{% endcomment %}
{%- assign me = "S. Chandak" -%}
{%- capture me_bold %}<span class="pub-me">{{ me }}</span>{% endcapture -%}

_**Google Scholar page [HERE](https://scholar.google.com/citations?user=czi8jdYAAAAJ&hl=en).**_

<div class="pub-page" markdown="0">

<ul>
  <li><a href="#papers">Papers</a>
    <ul>
      {%- for section in site.data.publications %}
      <li><a href="#{{ section.id }}">{{ section.toc | default: section.section | escape }}</a>
        {%- assign subs = section.papers | where_exp: "p", "p.subheading" %}
        {%- if subs.size > 0 %}
        <ul>
          {%- for sub in subs %}
          <li><a href="#{{ section.id }}-{{ sub.subheading | slugify }}">{{ sub.subheading | escape }}</a></li>
          {%- endfor %}
        </ul>
        {%- endif %}
      </li>
      {%- endfor %}
    </ul>
  </li>
  {%- if site.data.thesis %}
  <li><a href="#thesis">Ph.D. Thesis</a></li>
  {%- endif %}
  <li><a href="#talks">Invited Talks</a></li>
</ul>

<h2 id="papers">Papers</h2>
{% for section in site.data.publications %}
<h3 id="{{ section.id }}">{{ section.section | escape }}</h3>
{%- assign list_open = false %}
  {%- for paper in section.papers %}
  {%- if paper.subheading %}
  {%- if list_open %}
</ul>
  {%- assign list_open = false %}
  {%- endif %}
<h4 class="pub-subheading" id="{{ section.id }}-{{ paper.subheading | slugify }}">{{ paper.subheading | escape }}</h4>
  {%- continue %}
  {%- endif %}
  {%- unless list_open %}
<ul class="pub-list">
  {%- assign list_open = true %}
  {%- endunless %}
  <li>
    <p><span class="pub-title">{{ paper.title | escape }}</span><br />
    {{ paper.authors | escape | replace: me, me_bold }}<br />
    <em>{{ paper.venue | escape }}</em>
    {%- if paper.links %}<br />
    {% include pub-links.html links=paper.links %}
    {%- endif %}</p>
  </li>
  {%- endfor %}
  {%- if list_open %}
</ul>
  {%- endif %}
{% endfor %}

{% if site.data.thesis %}
<h2 id="thesis">Ph.D. Thesis</h2>

<ul class="pub-list">
  {%- for thesis in site.data.thesis %}
  <li>
    <p><span class="pub-title">{{ thesis.title | escape }}</span><br />
    {%- if thesis.authors %}
    {{ thesis.authors | escape | replace: me, me_bold }}<br />
    {%- endif %}
    <em>{{ thesis.venue | escape }}</em>
    {%- if thesis.links %}<br />
    {% include pub-links.html links=thesis.links %}
    {%- endif %}</p>
  </li>
  {%- endfor %}
</ul>
{% endif %}

<h2 id="talks">Invited Talks</h2>

<ul class="pub-talks">
  {%- for talk in site.data.talks %}
  <li>
    <p><span class="pub-title">{{ talk.title | escape }}</span></p>
    <ul>
      {%- for venue in talk.venues %}
      <li><em>{{ venue.name | escape }}</em>{% if venue.date %}, {{ venue.date | escape }}{% endif %}</li>
      {%- endfor %}
    </ul>
    {%- if talk.links %}
    <p>{% include pub-links.html links=talk.links %}</p>
    {%- endif %}
  </li>
  {%- endfor %}
</ul>

</div>
