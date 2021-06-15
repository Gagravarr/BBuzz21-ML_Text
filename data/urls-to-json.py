#!/usr/bin/python
# Fetches session pages, grabs the interesting bits, and saves as JSON
# Works for the older style site for 2015-2019, and the new online one
#  for 2020+
# For newer sites, use extract-urls.py to get the URLs. Older sites just
#  use grep + awk

import os,sys
import json
import urllib2
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

if len(sys.argv) < 2:
   print "Use:"
   print "  to-json.py <file-of-urls>"
   sys.exit(1)

output = sys.argv[2] if len(sys.argv) > 2 else "%s.json" % sys.argv[1]

# Grab the URLs from the file
urls = []
with open(sys.argv[1]) as inp:
   for line in inp:
      urls.append( line.strip() )
print "Found %d URLs to process" % len(urls)

# Shorter BS fetching
def div_class(html, cls):
   return html.find("div", attrs={"class":cls})
def content_span_text(html, cls):
   return text( html.find("span", attrs={"class":cls}) )
def text(html):
   if html:
      return html.text.strip()
   return ""

# Process them
talks = []
for url in urls:
   print " - %s" % url
   page = urllib2.urlopen(url)
   soup = BeautifulSoup(page, "lxml")

   # Get the main body bit - 2015-2019
   content = soup.body.find("section", attrs={"id":"main-content"})
   if content:
      # Get the talk title
      title = content.find("h1").text.strip()

      # Grab the content block
      content = div_class(content, "node-content")

      # Get the metadata we want
      speaker = text(div_class(content, "field-name-field-session-speaker"))
      level   = text(div_class(content, "field-name-field-session-exp-level-ref"))
      track   = text(div_class(content, "field-name-field-session-track-ref"))

      # Get the talk abstract
      abstr = content.find("section", attrs={"class":"field-name-field-session-description"}).find("div",attrs={"class":"field-items"}).text
   else:
      content = soup.body.find("main", attrs={"class":"block-main-content"})

      # This is very hacky and brittle...
      track = text(div_class(content, "field-node-field-track"))
      title = content_span_text(content, "field-name-title")
      abstr = text(div_class(content, "field-node--field-description"))
      ttype = text(div_class(content, "field-entity-reference-type-taxonomy-term"))
      level = text(div_class(content, "field-node-field-experience"))
      speaker = text(content.find("a", attrs={"class":"username"}))

      if track and title and abstr and ttype and speaker:
         pass
      else:
         print "Missing some"
         print track
         print title
         print ttype
         print level
         print speaker
         print abstr
         raise "Data Missing, please check html classes"
      if not level:
         level = "All"

   # Record
   talks.append({
      "title":title, "speaker":speaker, "level":level, "track":track,
      "abstract":abstr, "url": url
   })

# Save as JSON
print ""
with open(output, 'w') as outfile:
   json.dump(talks, outfile)
print "Saved to %s" % output
