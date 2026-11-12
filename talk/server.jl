# The deck, and any live-demo routes it needs.
#   julia --project=talk --threads=auto talk/server.jl    # serve on localhost:8081 and open it
using HTTP

# Demo routes: path => request -> response body (JSON). Add a live demo here, and a
# js/demo.js on its slide that fetches from it.
const ROUTES = Dict{String,Function}()

const TYPES = Dict(".html"=>"text/html; charset=utf-8", ".css"=>"text/css", ".js"=>"text/javascript", ".json"=>"application/json",
                   ".png"=>"image/png", ".jpg"=>"image/jpeg", ".svg"=>"image/svg+xml", ".woff2"=>"font/woff2",
                   ".mp4"=>"video/mp4", ".webm"=>"video/webm", ".gif"=>"image/gif")
# A file of the deck, with HTTP Range support so trimmed videos (data-start/data-end) can seek
function deckfile(req, path)
    f = normpath(joinpath(@__DIR__, lstrip(HTTP.unescapeuri(path), '/')))
    f = isdir(f) ? joinpath(f, "index.html") : f
    startswith(f, joinpath(@__DIR__, "")) && isfile(f) || return HTTP.Response(404)
    data = read(f); n = length(data)
    head = ["Content-Type"=>get(TYPES, splitext(f)[2], "application/octet-stream"), "Cache-Control"=>"no-cache", "Accept-Ranges"=>"bytes"]
    m = match(r"^bytes=(\d*)-(\d*)$", HTTP.header(req, "Range", ""))
    m === nothing && return HTTP.Response(200, head, data)
    a = isempty(m[1]) ? max(0, n-parse(Int, m[2])) : parse(Int, m[1])
    b = isempty(m[1]) || isempty(m[2]) ? n-1 : min(parse(Int, m[2]), n-1)
    a ≤ b || return HTTP.Response(416, ["Content-Range"=>"bytes */$n"])
    HTTP.Response(206, [head; "Content-Range"=>"bytes $a-$b/$n"], data[a+1:b+1])
end

function handle(req)
    u = HTTP.URI(req.target)
    route = get(ROUTES, u.path, nothing)
    route === nothing && return deckfile(req, u.path)
    HTTP.Response(200, ["Content-Type"=>"application/json"], route(req))
end

url = "http://localhost:8081"
server = HTTP.serve!(handle, "127.0.0.1", 8081)
println("Serving the talk on $url with $(Threads.nthreads()) threads")
try    # in the default browser; without one, the printed link will do
    run(Sys.iswindows() ? `cmd /c start $url` : Sys.isapple() ? `open $url` : `xdg-open $url`)
catch
end
wait(server)
