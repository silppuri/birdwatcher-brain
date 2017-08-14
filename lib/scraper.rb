require "pry"
require 'uri'
require 'csv'
require "nokogiri"
require "open-uri"

csv = CSV.open("output.csv", 'w',{:col_sep => ",", :quote_char => '\'', :force_quotes => true})
QUERY="http://www.xeno-canto.org/explore?query=box%3A58.124%2C3.955%2C71.581%2C34.014%20"
PAGE_QUERY_PARAM="pg"

def parse_header(header, csv)
  tarray = []
  tarray << "url"
  tarray << "common name"
  tarray << "scientific name"
  header.css("th")[2..-4].each do |col|
    tarray << col.text.strip
  end
  csv << tarray
end

def parse_results(results, csv)
  results.each do |row|
    tarray = []
    tarray << row.css(".jp-type-single")[0]["data-xc-filepath"]
    name, scientific_name = row.css("td")[1].text.strip.split(/(?=\()/)
    tarray << name.chop
    begin
      tarray << scientific_name&.delete("(").delete(")")
    rescue
      next
    end
    row.css("td")[2..-4].each do |col|
      tarray << col.text.strip
    end
    csv << tarray
  end
end

total_results = 0
doc = Nokogiri::HTML(open(QUERY))
puts "--- Page number: 1 ---"
puts "--- URI: #{QUERY} ---"
pages_count = doc.css(".results-pages li:nth-last-child(2)").first.text.to_i
header, *results = doc.css(".results tr")
parse_header(header, csv)
parse_results(results, csv)
total_results = total_results + results.length
puts "--- Found #{results.length} results ---"
sleep 1

for page_num in 276..pages_count do
  uri = "#{QUERY}&#{URI.encode_www_form([[PAGE_QUERY_PARAM, page_num]])}"
  puts "--- Page number: #{page_num} ---"
  puts "--- URI: #{uri} ---"
  doc = Nokogiri::HTML(open(uri))
  header, *results = doc.css(".results tr")
  puts "--- Found #{results.length} results ---"
  parse_results(results, csv)
  total_results = total_results + results.length
  sleep 1
end

puts "--- Total results count #{total_results} ---"
