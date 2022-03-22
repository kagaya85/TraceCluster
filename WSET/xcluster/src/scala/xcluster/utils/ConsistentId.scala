/* Copyright (C) 2017 University of Massachusetts Amherst.
   This file is part of “xcluster”
   http://github.com/iesl/xcluster
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package xcluster.utils

import java.util.UUID

/**
  * Generator of ids in deterministic way.
  * Note that this is NOT thread safe.
  */
object ConsistentId {
  var idno = 0

  def nextId = {
    val id = UUID.nameUUIDFromBytes(idno.toString.getBytes)
    idno += 1
    id
  }
}