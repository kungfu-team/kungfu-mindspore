#!/usr/bin/env ruby

for line in `cat log/127.0.0.1.40000@0.stdout.log |grep -o  value.* | awk '{print $2}' | awk -F ')' '{print $1}'`.each_line
  line.strip!
  puts line
end
