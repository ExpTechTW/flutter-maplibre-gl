// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

part of '../maplibre_gl.dart';

const _globalChannel = MethodChannel('plugins.flutter.io/maplibre_gl');

/// Retains active offline-download event-channel subscriptions so that Dart's
/// GC cannot tear them down during idle periods (e.g. while a download is
/// paused). If the subscription is collected, the native `EventSink` is
/// released and subsequent progress/success events are silently dropped,
/// causing the UI to appear stuck after resume.
final Map<String, StreamSubscription<dynamic>> _offlineDownloadSubscriptions =
    {};

/// Copy tiles db file passed in to the tiles cache directory (sideloaded) to
/// make tiles available offline.
Future<void> installOfflineMapTiles(String tilesDb) async {
  await _globalChannel.invokeMethod(
    'installOfflineMapTiles',
    <String, dynamic>{
      'tilesdb': tilesDb,
    },
  );
}

enum DragEventType { start, drag, end }

enum HoverEventType { enter, move, leave }

Future<dynamic> setOffline(bool offline) => _globalChannel.invokeMethod(
  'setOffline',
  <String, dynamic>{
    'offline': offline,
  },
);

/// Sets HTTP headers that are injected into every network request MapLibre
/// makes (tile fetches, style JSON, sprites, glyphs).
///
/// This is a **global** API: the headers apply to all map instances in the
/// process. To scope headers to specific URLs or to a single map instance,
/// use [MaplibreMapController.setCustomHeaders] instead.
///
/// Headers are applied at request time, so this method can be called before
/// or after a [MaplibreMap] widget is created. Calling it again replaces the
/// previously set headers; pass an empty map to clear all headers.
///
/// Example — attach a bearer token before showing any map:
/// ```dart
/// await setHttpHeaders({'Authorization': 'Bearer $token'});
/// ```
///
/// Example — attach multiple headers:
/// ```dart
/// await setHttpHeaders({
///   'Authorization': 'Bearer $token',
///   'X-Request-ID': requestId,
/// });
/// ```
///
/// Example — clear all previously set headers:
/// ```dart
/// await setHttpHeaders({});
/// ```
Future<void> setHttpHeaders(Map<String, String> headers) {
  return _globalChannel.invokeMethod(
    'setHttpHeaders',
    <String, dynamic>{
      'headers': headers,
    },
  );
}

const _tileCacheChannel = MethodChannel('plugins.flutter.io/maplibre_gl/tile_cache');

/// One cached tile body plus the response metadata needed to replay it.
///
/// The unit of exchange in both directions: Dart answers a lookup with these,
/// and pushes them ahead of demand via [injectMapLibreTiles].
class MapLibreTile {
  const MapLibreTile({
    required this.url,
    required this.data,
    this.contentType,
    this.etag,
  });

  final String url;
  final Uint8List data;
  final String? contentType;
  final String? etag;

  Map<String, Object?> _toWire() => <String, Object?>{
    'url': url,
    'data': data,
    if (contentType != null) 'contentType': contentType,
    if (etag != null) 'etag': etag,
  };
}

/// Called by native when MapLibre needs tile bodies it does not hold.
///
/// Requests arrive **batched** — native coalesces the tiles a viewport asks for
/// into one call — so the implementation should answer with a single store
/// lookup. Return only the URLs that hit; anything absent is treated as a miss
/// and fetched from the network.
typedef MapLibreTileCacheGetBatch =
    Future<List<MapLibreTile>> Function(List<String> urls);

/// Called by native after live tile downloads so Dart can persist + meter.
/// Also batched.
typedef MapLibreTileCachePutBatch =
    Future<void> Function(List<MapLibreTile> tiles);

/// Binds Dart as the tile cache authority for MapLibre HTTPS intercepts.
///
/// Native keeps only a bounded in-process mirror and never blocks a loader
/// thread waiting on Flutter — persistence, eviction policy, and
/// traffic metering all stay in Dart.
Future<void> bindMapLibreTileCache({
  required List<String> cacheablePatterns,
  required MapLibreTileCacheGetBatch getBatch,
  required MapLibreTileCachePutBatch putBatch,
}) async {
  _tileCacheChannel.setMethodCallHandler((call) async {
    switch (call.method) {
      // Native **pulls** this when it attaches. Binding usually happens during
      // app bootstrap, long before the first map exists and therefore before
      // this plugin is registered, so a push alone can land on a channel with
      // no handler and be lost — leaving native caching nothing, silently.
      case 'cacheablePatterns':
        return cacheablePatterns;
      case 'getBatch':
        final args = Map<String, Object?>.from(call.arguments as Map);
        final urls = (args['urls'] as List).cast<String>();
        final hits = await getBatch(urls);
        return <String, Object?>{
          for (final hit in hits)
            hit.url: <String, Object?>{
              'data': hit.data,
              'contentType': hit.contentType,
              'etag': hit.etag,
            },
        };
      case 'putBatch':
        final args = Map<String, Object?>.from(call.arguments as Map);
        final entries = (args['entries'] as List).cast<Object?>();
        await putBatch([
          for (final entry in entries)
            if (_tileFromWire(entry) case final tile?) tile,
        ]);
        return null;
      default:
        throw MissingPluginException(call.method);
    }
  });
  // Fast path when the plugin is already attached; the pull above is what makes
  // the ordering irrelevant.
  await _setCacheablePatterns(cacheablePatterns);
}

MapLibreTile? _tileFromWire(Object? entry) {
  if (entry is! Map) return null;
  final row = Map<Object?, Object?>.from(entry);
  final url = row['url'];
  final raw = row['data'];
  if (url is! String) return null;
  final Uint8List? data = switch (raw) {
    Uint8List u => u,
    List<int> l => Uint8List.fromList(l),
    _ => null,
  };
  if (data == null) return null;
  return MapLibreTile(
    url: url,
    data: data,
    contentType: row['contentType'] as String?,
    etag: row['etag'] as String?,
  );
}

/// Pushes [tiles] into native's in-process mirror so the next request for them
/// resolves with **no IPC and no network**.
///
/// This is the warm path: a timeline that knows which frame the finger is
/// heading for injects its tiles before revealing it, turning a scrub into
/// pure compositing. Pair with [mapLibreTilesMissing] to avoid re-sending
/// bytes native already holds.
Future<void> injectMapLibreTiles(List<MapLibreTile> tiles) async {
  if (tiles.isEmpty) return;
  try {
    await _tileCacheChannel.invokeMethod<void>('injectTiles', {
      'entries': [for (final tile in tiles) tile._toWire()],
    });
  } catch (_) {
    // Plugin not attached / binding not ready (tests, early bootstrap).
  }
}

/// The subset of [urls] native does **not** currently hold in memory.
///
/// Cheap (strings only) next to shipping tile bodies, so callers filter with
/// this before [injectMapLibreTiles].
Future<List<String>> mapLibreTilesMissing(List<String> urls) async {
  if (urls.isEmpty) return const [];
  try {
    final missing = await _tileCacheChannel.invokeMethod<List<Object?>>(
      'filterMissing',
      {'urls': urls},
    );
    return missing?.cast<String>() ?? urls;
  } catch (_) {
    return urls;
  }
}

/// Pushes the cacheable-URL substrings to native.
///
/// Private on purpose: patterns are declared once, through
/// [bindMapLibreTileCache], so a caller cannot bind handlers and then forget to
/// say what they cover.
Future<void> _setCacheablePatterns(List<String> patterns) async {
  try {
    await _tileCacheChannel.invokeMethod<void>('setCacheablePatterns', {
      'patterns': patterns,
    });
  } catch (_) {
    // Plugin not attached / binding not ready (tests, early bootstrap).
  }
}

/// Caps native's in-process tile mirror at [bytes] (LRU beyond that).
///
/// This is a *memory* budget in front of the Dart store, unrelated to
/// MapLibre's own ambient/offline database.
Future<void> setMapLibreTileMemoryLimit(int bytes) async {
  try {
    await _tileCacheChannel.invokeMethod<void>('setMemoryLimit', {
      'bytes': bytes,
    });
  } catch (_) {
    // Plugin not attached / binding not ready (tests, early bootstrap).
  }
}

/// Drops tiles whose URL contains one of [urlContains] from native's mirror.
Future<void> evictMapLibreTiles(List<String> urlContains) async {
  try {
    await _tileCacheChannel.invokeMethod<void>('evictTiles', {
      'contains': urlContains,
    });
  } catch (_) {
    // Plugin not attached / binding not ready (tests, early bootstrap).
  }
}

/// Cancels in-flight native tile HTTP (URLSession / OkHttp).
///
/// Pass [urlContains] to scope the cancel to abandoned frames — a scrub must
/// not take the basemap and the frame the finger landed on down with it. An
/// empty list cancels everything. Best-effort; safe anytime. Resolves once the
/// cancels have been applied.
Future<void> cancelMapLibreTileFetches({
  List<String> urlContains = const [],
}) async {
  try {
    await _tileCacheChannel.invokeMethod<void>('cancelPendingFetches', {
      'contains': urlContains,
    });
  } catch (_) {
    // Plugin not attached / binding not ready (tests, early bootstrap).
  }
}

Future<List<OfflineRegion>> mergeOfflineRegions(String path) async {
  final String regionsJson = await _globalChannel.invokeMethod(
    'mergeOfflineRegions',
    <String, dynamic>{
      'path': path,
    },
  );
  final regions = List<Map<String, dynamic>>.from(json.decode(regionsJson));
  return regions.map(OfflineRegion.fromMap).toList();
}

Future<List<OfflineRegion>> getListOfRegions() async {
  final String regionsJson = await _globalChannel.invokeMethod(
    'getListOfRegions',
    <String, dynamic>{},
  );
  final regions = List<Map<String, dynamic>>.from(json.decode(regionsJson));
  return regions.map(OfflineRegion.fromMap).toList();
}

Future<OfflineRegion> updateOfflineRegionMetadata(
  int id,
  Map<String, dynamic> metadata,
) async {
  final regionJson = await _globalChannel.invokeMethod(
    'updateOfflineRegionMetadata',
    <String, dynamic>{
      'id': id,
      'metadata': metadata,
    },
  );

  return OfflineRegion.fromMap(json.decode(regionJson));
}

Future<dynamic> setOfflineTileCountLimit(int limit) =>
    _globalChannel.invokeMethod(
      'setOfflineTileCountLimit',
      <String, dynamic>{
        'limit': limit,
      },
    );

/// Sets the maximum number of concurrent HTTP requests for tile downloads.
///
/// [maxRequests] controls the total number of concurrent requests (Android
/// only). [maxRequestsPerHost] controls the per-host concurrency limit
/// (both platforms). Lowering these values can help avoid rate limiting from
/// tile servers (e.g. Cloudflare).
///
/// Both values must be >= 1 when provided; OkHttp's `Dispatcher` throws on
/// non-positive values.
Future<void> setOfflineMaxConcurrentRequests({
  int? maxRequests,
  int? maxRequestsPerHost,
}) {
  assert(
    maxRequests == null || maxRequests >= 1,
    'maxRequests must be >= 1 (got $maxRequests)',
  );
  assert(
    maxRequestsPerHost == null || maxRequestsPerHost >= 1,
    'maxRequestsPerHost must be >= 1 (got $maxRequestsPerHost)',
  );
  return _globalChannel.invokeMethod(
    'setOfflineMaxConcurrentRequests',
    <String, dynamic>{
      if (maxRequests != null) 'maxRequests': maxRequests,
      if (maxRequestsPerHost != null) 'maxRequestsPerHost': maxRequestsPerHost,
    },
  );
}

/// Pauses an in-progress offline region download.
Future<void> pauseOfflineRegionDownload(int id) => _globalChannel.invokeMethod(
  'pauseOfflineRegionDownload',
  <String, dynamic>{'id': id},
);

/// Resumes a paused offline region download.
Future<void> resumeOfflineRegionDownload(int id) => _globalChannel.invokeMethod(
  'resumeOfflineRegionDownload',
  <String, dynamic>{'id': id},
);

/// Gets the current download status of an offline region.
Future<OfflineRegionStatus> getOfflineRegionStatus(int id) async {
  final String result = await _globalChannel.invokeMethod(
    'getOfflineRegionStatus',
    <String, dynamic>{'id': id},
  );
  return OfflineRegionStatus.fromMap(
    Map<String, dynamic>.from(json.decode(result)),
  );
}

Future<dynamic> deleteOfflineRegion(int id) => _globalChannel.invokeMethod(
  'deleteOfflineRegion',
  <String, dynamic>{
    'id': id,
  },
);

/// Removes all tiles from the shared ambient cache that are not associated
/// with any offline region. Call this after [deleteOfflineRegion] to fully
/// evict tiles that would otherwise be reused by a future download of the
/// same area.
Future<void> clearAmbientCache() =>
    _globalChannel.invokeMethod('clearAmbientCache');

/// Resets the entire offline database: deletes every offline region and
/// clears the ambient cache. Use with care — offline regions cannot be
/// recovered afterwards.
Future<void> resetOfflineDatabase() async {
  try {
    await _globalChannel.invokeMethod('resetOfflineDatabase');
  } finally {
    // Belt-and-suspenders: in the normal path, native emits a terminal
    // error for every in-flight download and cleanup() in the listener
    // has already removed the entry from the map. The short delay gives
    // any queued EventChannel messages a chance to be processed before we
    // forcibly cancel anything that somehow slipped through, guaranteeing
    // we don't leak retained subscriptions if the native terminal event
    // is ever missed.
    await Future<void>.delayed(const Duration(milliseconds: 50));
    final stragglers = _offlineDownloadSubscriptions.values.toList();
    _offlineDownloadSubscriptions.clear();
    for (final sub in stragglers) {
      unawaited(sub.cancel());
    }
  }
}

Future<OfflineRegion> downloadOfflineRegion(
  OfflineRegionDefinition definition, {
  Map<String, dynamic> metadata = const {},
  Function(DownloadRegionStatus event)? onEvent,
}) async {
  final channelName =
      'downloadOfflineRegion_${DateTime.now().microsecondsSinceEpoch}';

  await _globalChannel.invokeMethod(
    'downloadOfflineRegion#setup',
    <String, dynamic>{
      'channelName': channelName,
    },
  );

  if (onEvent != null) {
    void cleanup() {
      final sub = _offlineDownloadSubscriptions.remove(channelName);
      if (sub != null) unawaited(sub.cancel());
    }

    // Subscription is retained in _offlineDownloadSubscriptions and cancelled
    // in cleanup() on the terminal Success/Error event.
    // ignore: cancel_subscriptions
    final subscription = EventChannel(channelName)
        .receiveBroadcastStream()
        .handleError((error) {
          if (error is PlatformException) {
            onEvent(Error(error));
            cleanup();
            return Error(error);
          }
          final unknownError = Error(
            PlatformException(
              code: 'UnknowException',
              message:
                  'This error is unhandled by plugin. Please contact us if needed.',
              details: error,
            ),
          );
          onEvent(unknownError);
          cleanup();
          return unknownError;
        })
        .listen((data) {
          final Map<String, Object?> jsonData = json.decode(data);
          final status = switch (jsonData['status']) {
            'start' => InProgress(0.0),
            'progress' => InProgress(
              (jsonData['progress']! as num).toDouble(),
              completedResourceCount:
                  (jsonData['completedResourceCount'] as num?)?.toInt() ?? 0,
              requiredResourceCount:
                  (jsonData['requiredResourceCount'] as num?)?.toInt() ?? 0,
              completedResourceSize:
                  (jsonData['completedResourceSize'] as num?)?.toInt() ?? 0,
            ),
            'success' => Success(),
            _ => throw Exception('Invalid event status ${jsonData['status']}'),
          };
          onEvent(status);
          if (status is Success) cleanup();
        });

    _offlineDownloadSubscriptions[channelName] = subscription;
  }

  final result = await _globalChannel.invokeMethod(
    'downloadOfflineRegion',
    <String, dynamic>{
      'definition': definition.toMap(),
      'metadata': metadata,
    },
  );

  return OfflineRegion.fromMap(json.decode(result));
}
