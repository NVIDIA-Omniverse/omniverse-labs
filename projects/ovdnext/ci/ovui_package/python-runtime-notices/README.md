# Python dependency notices

Both pinned install-only inputs omit dependency notices. Packaging restores
hash-pinned texts under `python/licenses/`, verifies them before output and after
copying, and records their sources and measured bytes. Fresh downloaded-package
verification checks the complete provenance and delivered notice bytes on both
platforms. `ci/ovui-python-3.12.16.json` binds these resources to the unchanged
platform-specific interpreter input.

The current CPython 3.12.16 core license remains at `python/LICENSE.txt` on
Windows and `python/lib/python3.12/LICENSE.txt` on Linux, with another copy at
`app/ovd_ui/notices/Python-LICENSE.txt`. The Windows input license also retains
its Bzip2 1.0.8 and Microsoft notices.

| Additional notice | Windows consumer | Linux consumer |
| --- | --- | --- |
| LICENSE.cpython.txt | Exact 3.12.16 incorporated-software acknowledgments, including libmpdec 2.5.1 | PBS-maintained incorporated-software acknowledgments |
| LICENSE.bzip2.txt | Covered by the existing input license | _bz2 |
| LICENSE.libedit.txt | — | readline |
| LICENSE.libffi.txt | _ctypes | _ctypes |
| LICENSE.liblzma.txt | _lzma | _lzma |
| LICENSE.libuuid.txt | — | _uuid |
| LICENSE.mpdecimal.txt | Covered by the incorporated-software acknowledgments | _decimal |
| LICENSE.ncurses.txt | — | _curses, _curses_panel, readline |
| LICENSE.openssl-3.txt | _hashlib, _ssl | _hashlib, _ssl |
| LICENSE.sqlite.txt | _sqlite3 | _sqlite3 |
| LICENSE.zlib.txt | zlib, binascii | zlib, binascii |

Linux restores eleven notices. Bzip2 uses the official 1.0.8 source LICENSE
and its 2019 copyright; zlib uses the exact 1.3.2 source LICENSE and its
1995-2026 copyright, matching Windows. Their source archive and notice hashes,
and the superseded generic PBS hashes, remain in the manifest. The other nine
texts come from the verified full PBS archive. These are PBS-maintained
notices, rather than extracts from each linked library release; their
copyright dates can predate the delivered versions. The exact current
CPython core license is the separate file identified above.

The Linux full-archive inventory contains twenty texts. Nine are excluded:
Berkeley DB (`_dbm` is not delivered), X11/Xau/XCB/Tcl/Tix consumers,
unused zlib-ng/zstd entries, and the generic Expat notice. Expat 2.9.0's exact
notice is retained separately at `app/ovd_ui/notices/Expat-LICENSE.txt`.
OpenSSL 3.5.9 and zlib 1.3.2 are the actual CPython dependency versions;
alternative build-metadata entries do not establish linked versions.

The verified Windows full PBS archive contains the same generic twenty-text
PBS notice set as the Linux full archive, with CRLF rather than LF line endings.
Windows instead restores six exact texts from pinned upstream source archives
or commits, matching its shipped dependencies. The manifest's `source_archive`
and `source_metadata_sha256` identify the interpreter's upstream build;
`cpython_source` and `dependency_sources` identify the sources of the restored
Windows notice texts. The CPython notice is the exact
3.12.16 `Doc/license.rst`, whose incorporated-software section is absent from
the install-only input's `LICENSE.txt`. Windows libffi uses the exact
`cpython-source-deps` commit selected by the PBS Windows build, rather than
the Linux libffi release. Source and notice hashes are recorded per component.

The compiled Python hash implementations retain their HACL* MIT and KaRaMeL
Apache 2.0 notices at `app/ovd_ui/notices/HACL-LICENSE.txt` on both platforms.
The resource combines the first license comments from the pinned CPython
3.12.16 `Modules/_hacl/Hacl_Hash_MD5.c` and
`Modules/_hacl/include/krml/FStar_UInt_8_16_32_64.h` with the full Apache terms
from OpenSSL 3.5.9 `LICENSE.txt`. The Windows manifest records both source
archives; these headers represent the copyright/license notices in the
compiled HACL* and KaRaMeL include graph. Packaging and fresh inspection
enforce the combined resource SHA-256
`3a2394d9022fc168845231b08e98a6eade120257465968655cc4d79b7e9828a9`.

Preserve intentional upstream whitespace: Git disables text conversion for
frozen platform notice resources. Notice retention and verified provenance
do not establish OSRB, export or customer redistribution approval.
