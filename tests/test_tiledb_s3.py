"""Exercise native TileDB S3 access against a local, disposable S3 server."""

import multiprocessing

import boto3
import numpy as np
import pytest
from moto.server import ThreadedMotoServer
from pyproj import CRS

from pyforestscan.handlers import convert_to_tiledb, read_lidar, write_tiledb
from pyforestscan.tiledb_io import get_tiledb_info


def _serve_s3(connection):
    # Native TileDB calls can hold the GIL while waiting for HTTP responses.
    # A separate process lets the emulator answer those requests.
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        connection.send(server.get_host_and_port())
        connection.recv()
    finally:
        server.stop()
        connection.close()


@pytest.fixture
def s3_array(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve_s3, args=(child,))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "Local S3 emulator did not start"
        host, port = parent.recv()
        client = boto3.client(
            "s3", endpoint_url=f"http://{host}:{port}", region_name="us-east-1",
            aws_access_key_id="test", aws_secret_access_key="test",
        )
        client.create_bucket(Bucket="pyforestscan-test")
        config = tmp_path / "s3.config"
        config.write_text(
            f"vfs.s3.endpoint_override {host}:{port}\n"
            "vfs.s3.scheme http\n"
            "vfs.s3.region us-east-1\n"
            "vfs.s3.use_virtual_addressing false\n"
            "vfs.s3.aws_access_key_id test\n"
            "vfs.s3.aws_secret_access_key test\n"
            "vfs.s3.connect_timeout_ms 1000\n"
            "vfs.s3.request_timeout_ms 10000\n"
            "vfs.s3.connect_max_tries 1\n"
            "sm.compute_concurrency_level 2\n"
            "sm.io_concurrency_level 2\n"
        )
        yield "s3://pyforestscan-test/forest", str(config), client
    finally:
        if process.is_alive():
            parent.send("stop")
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)


def test_s3_write_metadata_and_bounded_read(s3_array):
    """Both Python metadata access and native PDAL I/O honor the config file."""
    uri, config, client = s3_array
    points = np.array([
        (500000.123456789, 2200000.1, 101.25, 4.0, 2),
        (500001.987654321, 2200001.2, 105.75, 8.5, 5),
        (500001.987654321, 2200001.2, 105.75, 9.0, 5),
    ], dtype=[
        ("X", "f8"), ("Y", "f8"), ("Z", "f8"),
        ("HeightAboveGround", "f8"), ("Classification", "u1"),
    ])
    write_tiledb(points, uri, "EPSG:32605", config_file=config, chunk_size=2)

    info = get_tiledb_info(uri, config_file=config)
    assert info["bounds"] == (
        points["X"].min(), points["X"].max(),
        points["Y"].min(), points["Y"].max(),
        points["Z"].min(), points["Z"].max(),
    )
    assert CRS.from_user_input(info["srs"]) == CRS.from_epsg(32605)

    arrays = read_lidar(
        uri, source_type="tiledb", tiledb_config=config,
        bounds=([500001, 500002], [2200000, 2200002]),
    )
    actual = np.concatenate(arrays)
    assert len(actual) == 2
    np.testing.assert_array_equal(actual["X"], points["X"][1:])
    np.testing.assert_array_equal(np.sort(actual["HeightAboveGround"]), [8.5, 9.0])
    np.testing.assert_array_equal(actual["Classification"], [5, 5])
    assert read_lidar(
        uri, source_type="tiledb", tiledb_config=config,
        bounds=([0, 1], [0, 1]),
    ) is None

    with pytest.raises(FileExistsError):
        write_tiledb(points, uri, "EPSG:32605", config_file=config)

    for key in ("occupied/sentinel", "an-object"):
        client.put_object(Bucket="pyforestscan-test", Key=key, Body=b"keep me")
        destination = "s3://pyforestscan-test/" + key.split("/")[0]
        with pytest.raises(FileExistsError):
            write_tiledb(points, destination, "EPSG:32605", config_file=config)
        stored = client.get_object(Bucket="pyforestscan-test", Key=key)
        assert stored["Body"].read() == b"keep me"

    copied = uri + "-copy"
    count = convert_to_tiledb(
        uri, copied, source_type="tiledb", config_file=config, chunk_size=2,
        bounds=([500001, 500002], [2200000, 2200002]),
    )
    assert count == 2
    actual = np.concatenate(read_lidar(copied, source_type="tiledb", tiledb_config=config))
    np.testing.assert_array_equal(actual["X"], points["X"][1:])
    np.testing.assert_array_equal(np.sort(actual["HeightAboveGround"]), [8.5, 9.0])
