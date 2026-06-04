<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

This file covers the open-source and third-party software used by the
`ov-libaries-livestream` and `usd-viewer-example` Python samples.  Each
entry names the package, the version pinned in `pyproject.toml`, the
applicable license, and the required attribution or notice text sourced
directly from the installed distribution.

---

## numpy 2.2.6

**License:** BSD 3-Clause  
**Homepage:** <https://numpy.org>

```
Copyright (c) 2005-2024, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the
      distribution.

    * Neither the name of the NumPy Developers nor the names of any
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The numpy binary distribution for Windows also bundles **OpenBLAS**
(BSD 3-Clause, Copyright © 2011-2014 The OpenBLAS Project) and **LAPACK**
(BSD 3-Clause-Attribution, Copyright © 1992-2013 The University of Tennessee
and others).  Full notices are included in `numpy-2.2.6.dist-info/LICENSE.txt`
inside the installed package.

---

## ovrtx 0.3.0.312915

**License:** NVIDIA Proprietary Software  
**Homepage:** <https://developer.nvidia.com/omniverse>

ovrtx is governed by the
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the
[NVIDIA Omniverse Product-Specific Terms](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-omniverse/).
By downloading or using ovrtx, you agree to those terms.

---

## ovphysx 0.3.7

**License:** LicenseRef-NVIDIA-Omniverse  
**Homepage:** <https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx>

```
ovphysx is governed by the following NVIDIA Agreements:

Enterprise Software | NVIDIA Software License Agreement and NVIDIA Agreements
https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/

and

NVIDIA Agreements | Enterprise Software | Product Specific Terms for Omniverse
https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-omniverse

By downloading or using ovphysx, you agree to the NVIDIA Omniverse terms.
```

Binary distributions of ovphysx may include third-party libraries covered by
their own licenses; these are documented in the `ovphysx-LICENSES.zip` file
distributed alongside the package.

---

## Pillow 12.2.0

**License:** MIT-CMU (Historical Permission Notice and Disclaimer)  
**Homepage:** <https://python-pillow.org>

```
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

The Pillow binary distribution for Windows bundles additional libraries
(FreeType, libjpeg-turbo, zlib, brotli, and others), each under their own
open-source licenses.  Full notices are included in
`pillow-12.2.0.dist-info/licenses/LICENSE` inside the installed package.

---

## PySide6 6.8.2.1 · pyside6-addons 6.8.2.1 · pyside6-essentials 6.8.2.1 · shiboken6 6.8.2.1

**License:** GNU Lesser General Public License v3 (LGPL-3.0) for the Python
bindings; the underlying Qt libraries are available under LGPL-3.0 or a
commercial Qt license.  
**Homepage:** <https://pyside.org>  
**LGPL-3.0 full text:** <https://www.gnu.org/licenses/lgpl-3.0.html>

PySide6 is the official Python bindings for the Qt framework, provided by
The Qt Company.  Under the LGPL, you may use PySide6 in your application
without being required to release your own source code, provided you meet
the LGPL's requirements for dynamic linking and the ability to re-link
against a modified version of the library.  See the Qt Company's
[open-source licensing FAQ](https://www.qt.io/licensing/open-source-lgpl-obligations)
for details.

---

## packaging 26.2

**License:** Apache-2.0 OR BSD-2-Clause  
**Homepage:** <https://packaging.pypa.io/>

Distributed under your choice of either license.

**BSD 2-Clause:**

```
Copyright (c) Donald Stufft and individual contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```

**Apache-2.0** full text: <https://www.apache.org/licenses/LICENSE-2.0>
