#!/usr/bin/env ruby

def extract_log(filename)
  `cat #{filename} |grep -o  value.* | awk '{print $2}' | awk -F ')' '{print $1}'`
end

def single
  for line in extract_log("log/127.0.0.1.40000@0.stdout.log").each_line
    line.strip!
    puts line
  end
end

def elastic
  for i in 0..9
    for line in extract_log("log/127.0.0.1.40000@#{i}.stdout.log").each_line
      line.strip!
      puts line
    end
  end
end

#single
elastic
