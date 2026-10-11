import Foundation
import MapLibre

public class MapLibreCustomHeaders {
    private static let queue = DispatchQueue(
        label: "io.flutter.maplibre_gl.customHeaders",
        attributes: .concurrent
    )
    private static var _customHeaders: [String: String] = [:]
    private static var _filterPatterns: [String] = []
    /// Compiled once per `setCustomHeaders`. `String.range(of:options:.regularExpression)`
    /// recompiles on every request, and every tile goes through here.
    private static var _compiledFilters: [CompiledFilter] = []

    private struct CompiledFilter {
        let source: String
        let regex: NSRegularExpression?

        func matches(_ url: String) -> Bool {
            if let regex {
                let length = (url as NSString).length
                return regex.firstMatch(
                    in: url,
                    options: [],
                    range: NSRange(location: 0, length: length)
                ) != nil
            }
            // Patterns NSRegularExpression rejects (an empty pattern matches
            // every string here; an invalid one matches nothing).
            return url.range(of: source, options: .regularExpression) != nil
        }
    }

    // Sets both headers and filter atomically.
    public static func setCustomHeaders(_ headers: [String: String], filter: [String]) {
        let compiled = filter.map {
            CompiledFilter(source: $0, regex: try? NSRegularExpression(pattern: $0))
        }
        queue.async(flags: .barrier) {
            _customHeaders = headers
            _filterPatterns = filter
            _compiledFilters = compiled
        }
    }

    // Updates only headers, leaving the existing filter intact.
    public static func setHeaders(_ headers: [String: String]) {
        queue.async(flags: .barrier) {
            _customHeaders = headers
        }
    }

    public static func getCustomHeaders() -> [String: String] {
        queue.sync { _customHeaders }
    }

    public static func getFilterPatterns() -> [String] {
        queue.sync { _filterPatterns }
    }

    // Whether there is anything to inject at all. Requests are only intercepted
    // when there is.
    public static func hasCustomHeaders() -> Bool {
        queue.sync { !_customHeaders.isEmpty }
    }

    // Returns headers and whether they should be applied, in a single lock acquisition.
    public static func headersIfApplicable(to url: String) -> (headers: [String: String], apply: Bool) {
        let (headers, filters) = queue.sync { (_customHeaders, _compiledFilters) }
        if filters.isEmpty {
            return (headers, true)
        }
        for filter in filters where filter.matches(url) {
            return (headers, true)
        }
        return (headers, false)
    }

    public static func shouldApplyHeaders(to url: String) -> Bool {
        headersIfApplicable(to: url).apply
    }
}
