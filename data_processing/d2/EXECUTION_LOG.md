# D2 mutation 执行日志

时间使用 Asia/Shanghai。只记录已经执行的步骤。

## 2026-11-06T16:55:05+08:00 · download · error

LFS object size 342402448 is outside the allowed range
Traceback (most recent call last):
  File "G:\LLLM\MCDHGN-main\data_processing\d2\pipeline.py", line 689, in main
    args.func(args)
  File "G:\LLLM\MCDHGN-main\data_processing\d2\pipeline.py", line 331, in cmd_download
    config, files = download_studies(
  File "G:\LLLM\MCDHGN-main\data_processing\d2\download.py", line 246, in download_studies
    receipt = download_relpath(relpath, dest, config, downloaded_at)
  File "G:\LLLM\MCDHGN-main\data_processing\d2\download.py", line 169, in download_relpath
    pointer = parse_pointer(raw)
  File "G:\LLLM\MCDHGN-main\data_processing\d1\download.py", line 82, in parse_pointer
    raise DownloadError(f"LFS object size {size} is outside the allowed range")
data_processing.d1.download.DownloadError: LFS object size 342402448 is outside the allowed range

## 2026-11-06T16:55:25+08:00 · download · error

Protected inputs changed: d1_readonly:data_versions/d1_cnv_v1/source_manifest.json
Traceback (most recent call last):
  File "G:\LLLM\MCDHGN-main\data_processing\d2\pipeline.py", line 689, in main
    args.func(args)
  File "G:\LLLM\MCDHGN-main\data_processing\d2\pipeline.py", line 329, in cmd_download
    verify_protected()
  File "G:\LLLM\MCDHGN-main\data_processing\d2\pipeline.py", line 97, in verify_protected
    raise RuntimeError("Protected inputs changed: " + ", ".join(mismatches[:30]))
RuntimeError: Protected inputs changed: d1_readonly:data_versions/d1_cnv_v1/source_manifest.json

## 2026-11-06T18:03:15+08:00 · download · error

The read operation timed out
y", line 277, in download_studies
    receipt = download_relpath(relpath, dest, config, downloaded_at)
  File "G:\LLLM\MCDHGN-main\data_processing\d2\download.py", line 198, in download_relpath
    raw = _read_bytes(RAW_ROOT + "/" + relpath)
  File "G:\LLLM\MCDHGN-main\data_processing\d2\download.py", line 73, in _read_bytes
    with _request(url, timeout=timeout) as response:
  File "G:\LLLM\MCDHGN-main\data_processing\d2\download.py", line 66, in _request
    return urllib.request.urlopen(req, timeout=timeout)
  File "G:\LLLM\MCDHGN-main\.conda\lib\urllib\request.py", line 214, in urlopen
    return opener.open(url, data, timeout)
  File "G:\LLLM\MCDHGN-main\.conda\lib\urllib\request.py", line 517, in open
    response = self._open(req, data)
  File "G:\LLLM\MCDHGN-main\.conda\lib\urllib\request.py", line 534, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "G:\LLLM\MCDHGN-main\.conda\lib\urllib\request.py", line 494, in _call_chain
    result = func(*args)
  File "G:\LLLM\MCDHGN-main\.conda\lib\urllib\request.py", line 1389, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "G:\LLLM\MCDHGN-main\.conda\lib\urllib\request.py", line 1350, in do_open
    r = h.getresponse()
  File "G:\LLLM\MCDHGN-main\.conda\lib\http\client.py", line 1377, in getresponse
    response.begin()
  File "G:\LLLM\MCDHGN-main\.conda\lib\http\client.py", line 320, in begin
    version, status, reason = self._read_status()
  File "G:\LLLM\MCDHGN-main\.conda\lib\http\client.py", line 281, in _read_status
    line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
  File "G:\LLLM\MCDHGN-main\.conda\lib\socket.py", line 704, in readinto
    return self._sock.recv_into(b)
  File "G:\LLLM\MCDHGN-main\.conda\lib\ssl.py", line 1242, in recv_into
    return self.read(nbytes, buffer)
  File "G:\LLLM\MCDHGN-main\.conda\lib\ssl.py", line 1100, in read
    return self._sslobj.read(len, buffer)
socket.timeout: The read operation timed out

## 2026-11-06T19:41:48+08:00 · download · ok

Downloaded/reused 120 files into G:\LLLM\MCDHGN-main\data_versions\d2_mutation_v1\sources

## 2026-11-06T19:44:59+08:00 · build · ok

{"n_genes": 12944, "observed_cells": 621072, "callable": {"KIRC": 402, "BRCA": 1066, "READ": 139, "PRAD": 494, "STAD": 436, "HNSC": 515, "LUAD": 566, "THCA": 489, "BLCA": 410, "ESCA": 182, "LIHC": 366, "UCEC": 517, "COAD": 394, "LUSC": 484, "CESC": 291, "KIRP": 276}}

## 2026-11-06T19:45:30+08:00 · seal · error

Protected file hash mismatches after D2: d1_readonly:data_versions/d1_cnv_v1/source_manifest.json
Traceback (most recent call last):
  File "G:\LLLM\MCDHGN-main\data_processing\d2\pipeline.py", line 705, in main
    args.func(args)
  File "G:\LLLM\MCDHGN-main\data_processing\d2\pipeline.py", line 685, in cmd_seal
    raise RuntimeError("Protected file hash mismatches after D2: " + ", ".join(after["mismatches"][:20]))
RuntimeError: Protected file hash mismatches after D2: d1_readonly:data_versions/d1_cnv_v1/source_manifest.json

## 2026-11-06T19:45:59+08:00 · seal · ok

{"hard_mismatches": 0, "soft_mismatches": ["d1_readonly:data_versions/d1_cnv_v1/source_manifest.json"]}

## 2026-11-06T19:46:51+08:00 · final_status · ok

- download: 15 studies / 120 files under data_versions/d2_mutation_v1/sources (DataHub commit 860b63a7); legacy GDC Mutect2 UUIDs unavailable
- build: mutation_type_features.tsv 12944 x 48; observed_cells=621072; callable patients per cancer recorded in quality_report.json
- tests: python -m unittest tests.test_d2 → 9 OK
- verify/seal: hard-protected hashes unchanged (122); soft note only for concurrent d1 source_manifest.json drift (not reverted)
- training not started; graph.bin not written; old mut_data not written

