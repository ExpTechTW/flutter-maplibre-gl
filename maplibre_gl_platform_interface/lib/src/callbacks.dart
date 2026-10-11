// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

part of '../maplibre_gl_platform_interface.dart';

/// Callback function taking a single argument.
typedef ArgumentCallback<T> = void Function(T argument);

/// Mutable collection of [ArgumentCallback] instances, itself an [ArgumentCallback].
///
/// Additions and removals happening during a single [call] invocation do not
/// change who gets a callback until the next such invocation.
///
/// Optimized for the singleton case.
class ArgumentCallbacks<T> {
  final List<ArgumentCallback<T>> _callbacks = <ArgumentCallback<T>>[];

  /// Fired after the listener list actually changes.
  ///
  /// Platform implementations use this to subscribe to a high-frequency native
  /// event only while somebody is listening. App code should use [add] and
  /// [remove].
  void Function()? onListenersChanged;

  /// Callback method. Invokes the corresponding method on each callback
  /// in this collection.
  ///
  /// The list of callbacks being invoked is computed at the start of the
  /// method and is unaffected by any changes subsequently made to this
  /// collection.
  void call(T argument) {
    final length = _callbacks.length;
    if (length == 1) {
      _callbacks[0].call(argument);
    } else if (0 < length) {
      for (final callback in List<ArgumentCallback<T>>.from(_callbacks)) {
        callback(argument);
      }
    }
  }

  /// Adds a callback to this collection.
  void add(ArgumentCallback<T> callback) {
    _callbacks.add(callback);
    onListenersChanged?.call();
  }

  /// Removes a callback from this collection.
  ///
  /// Does nothing, if the callback was not present.
  void remove(ArgumentCallback<T> callback) {
    if (_callbacks.remove(callback)) {
      onListenersChanged?.call();
    }
  }

  /// Removes all callbacks
  void clear() {
    if (_callbacks.isEmpty) return;
    _callbacks.clear();
    onListenersChanged?.call();
  }

  /// Whether this collection is empty.
  bool get isEmpty => _callbacks.isEmpty;

  /// Whether this collection is non-empty.
  bool get isNotEmpty => _callbacks.isNotEmpty;
}
