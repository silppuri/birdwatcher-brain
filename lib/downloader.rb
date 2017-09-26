require 'csv'
require 'open-uri'
require 'fileutils'

quote_char = "\'"
total_length = 0


def to_seconds(timestamp)
  timestamp.split(":").map(&:to_i).reduce(0) { |minutes, seconds| minutes * 60 + seconds }
end

def filepath(row)
  "#{row[2].gsub("\s", "_").downcase}/#{row[0].split("/").last}"
end

CSV.foreach('data/recordings.csv', headers: :first_row, quote_char: quote_char) do |row|
  total_length = total_length + to_seconds(row[3])
  dst = "./data/mp3/#{filepath(row)}"
  FileUtils.mkdir_p(File.dirname(dst))
  io_source = open(row[0])
  IO.copy_stream(io_source, dst)
end

puts total_length
