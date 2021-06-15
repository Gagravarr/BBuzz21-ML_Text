#!/usr/bin/python
# Grabs all the session URLs from a saved copy of the session list,
#  ready for processing by urls-to-json.py
# Only works on the newer online 2020+ style site
# Sessions page should be saved as sessions-<yy>.html

import os,sys
import json
import urllib2
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

if len(sys.argv) < 2:
   print "Use:"
   print "  to-json.py <yy>"
   sys.exit(1)

year = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else "%s/links-%s.txt" % (year,year)

inpfile = "sessions-%s.html" % year
base = "https://20%s.berlinbuzzwords.de" % year
sessions = []

# Process the file, finding the session links
with open(inpfile, "r") as inp:
   contents = inp.read()
   soup = BeautifulSoup(contents, "lxml")

   # Get the main body bit
   content = soup.body.find("div", attrs={"class":"view-content"})

   # Cheat, find all links, and ignore non-session ones
   atags = content.find_all("a")
   for a in atags:
      href = a["href"]
      if href and href.startswith("/session/"):
         sessions.append( base+href )
sessions.sort()

# Save list
print ""
with open(output, 'w') as outfile:
   for link in sessions:
      outfile.write(link)
      outfile.write("\n")
print "Saved %d sessions to %s" % (len(sessions), output)
