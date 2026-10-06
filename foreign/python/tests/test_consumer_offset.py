# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest

from apache_iggy import (
    Consumer,
    ConsumerOffsetInfo,
    HttpConfig,
    IggyClient,
    PollingStrategy,
)
from apache_iggy import SendMessage as Message

from .utils import get_http_server_config, wait_for_ping

MESSAGES_PER_PARTITION = 5


def _last_offset(partition_id: int) -> int:
    # Partitions hold different message counts so a `current_offset` read from
    # the wrong partition cannot match.
    return MESSAGES_PER_PARTITION + partition_id - 1


async def _create_topic_with_messages(
    iggy_client: IggyClient, unique_name, partitions_count: int = 1
) -> tuple[str, str]:
    stream_name = unique_name()
    topic_name = unique_name()

    await iggy_client.create_stream(stream_name)
    await iggy_client.create_topic(
        stream=stream_name, name=topic_name, partitions_count=partitions_count
    )
    for partition_id in range(partitions_count):
        await iggy_client.send_messages(
            stream=stream_name,
            topic=topic_name,
            partitioning=partition_id,
            messages=[
                Message(f"Partition {partition_id} message {index}")
                for index in range(_last_offset(partition_id) + 1)
            ],
        )
    return stream_name, topic_name


def _as_tuple(info: ConsumerOffsetInfo | None) -> tuple[int, int, int] | None:
    if info is None:
        return None
    return info.partition_id, info.current_offset, info.stored_offset


async def _call_offset_method(
    iggy_client: IggyClient,
    unique_name,
    method: str,
    *,
    consumer: object = None,
    partition_id: int = 0,
):
    kwargs = {
        "consumer": Consumer.Single(1) if consumer is None else consumer,
        "partition_id": partition_id,
    }
    if method == "store":
        kwargs["offset"] = 0
    return await getattr(iggy_client, f"{method}_consumer_offset")(
        unique_name(), unique_name(), **kwargs
    )


class TestStoreAndGetConsumerOffset:
    """Test storing an offset and reading it back."""

    @pytest.mark.asyncio
    async def test_get_consumer_offset_returns_stored_offset(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test the getters report the partition, newest offset and stored offset."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name, partitions_count=2
        )
        consumer = Consumer.Single(unique_name())

        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=consumer, offset=2, partition_id=1
        )
        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=consumer, partition_id=1
        )

        assert isinstance(info, ConsumerOffsetInfo)
        assert type(info.partition_id) is int
        assert type(info.current_offset) is int
        assert type(info.stored_offset) is int
        assert _as_tuple(info) == (1, _last_offset(1), 2)

    @pytest.mark.asyncio
    async def test_get_consumer_offset_without_a_stored_offset_returns_none(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a consumer that never stored an offset reads back `None`."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )

        info = await iggy_client.get_consumer_offset(
            stream_name,
            topic_name,
            consumer=Consumer.Single(unique_name()),
            partition_id=0,
        )

        assert info is None

    @pytest.mark.asyncio
    async def test_store_consumer_offset_overwrites_the_previous_offset(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a later store replaces the earlier one, moving backwards too."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        consumer = Consumer.Single(unique_name())

        for offset in (3, 1, _last_offset(0), 0):
            await iggy_client.store_consumer_offset(
                stream_name,
                topic_name,
                consumer=consumer,
                offset=offset,
                partition_id=0,
            )
            info = await iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=0
            )
            assert _as_tuple(info) == (0, _last_offset(0), offset)

    @pytest.mark.asyncio
    async def test_consumer_offsets_are_kept_per_consumer_and_partition(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a store touches only its own consumer and partition."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name, partitions_count=2
        )
        first = Consumer.Single(unique_name())
        second = Consumer.Single(unique_name())

        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=first, offset=1, partition_id=0
        )
        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=first, offset=3, partition_id=1
        )
        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=second, offset=2, partition_id=0
        )

        offsets = [
            _as_tuple(
                await iggy_client.get_consumer_offset(
                    stream_name, topic_name, consumer=consumer, partition_id=partition
                )
            )
            for consumer, partition in (
                (first, 0),
                (first, 1),
                (second, 0),
                (second, 1),
            )
        ]
        assert offsets == [
            (0, _last_offset(0), 1),
            (1, _last_offset(1), 3),
            (0, _last_offset(0), 2),
            None,
        ]

    @pytest.mark.asyncio
    async def test_consumer_offset_accepts_numeric_ids(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test numeric stream, topic and consumer ids address the same offset."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        stream = await iggy_client.get_stream(stream_name)
        assert stream is not None
        topic = await iggy_client.get_topic(stream.id, topic_name)
        assert topic is not None

        await iggy_client.store_consumer_offset(
            stream.id, topic.id, consumer=Consumer.Single(42), offset=3, partition_id=0
        )
        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=Consumer.Single(42), partition_id=0
        )

        assert _as_tuple(info) == (0, _last_offset(0), 3)

    @pytest.mark.asyncio
    async def test_get_consumer_offset_without_partition_id_reads_partition_zero(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test an omitted partition reads partition 0, not another stored one."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name, partitions_count=2
        )
        consumer = Consumer.Single(unique_name())
        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=consumer, offset=3, partition_id=1
        )
        assert (
            await iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=consumer
            )
            is None
        )

        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=consumer, offset=1, partition_id=0
        )
        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=consumer
        )

        assert _as_tuple(info) == (0, _last_offset(0), 1)

    @pytest.mark.asyncio
    async def test_stored_offset_moves_where_next_resumes(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a manual commit after an `auto_commit=False` poll moves `Next`."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        consumer = Consumer.Single(unique_name())

        batch = await iggy_client.poll_messages(
            stream_name,
            topic_name,
            consumer=consumer,
            partition_id=0,
            polling_strategy=PollingStrategy.Next(),
            count=2,
            auto_commit=False,
        )
        assert [message.offset() for message in batch] == [0, 1]
        assert (
            await iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=0
            )
            is None
        )

        await iggy_client.store_consumer_offset(
            stream_name,
            topic_name,
            consumer=consumer,
            offset=batch[-1].offset(),
            partition_id=0,
        )
        resumed = await iggy_client.poll_messages(
            stream_name,
            topic_name,
            consumer=consumer,
            partition_id=0,
            polling_strategy=PollingStrategy.Next(),
            count=10,
            auto_commit=False,
        )

        assert [message.offset() for message in resumed] == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_get_consumer_offset_reads_an_auto_committed_offset(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test the offset `poll_messages(auto_commit=True)` stores is readable."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        consumer = Consumer.Single(unique_name())

        await iggy_client.poll_messages(
            stream_name,
            topic_name,
            consumer=consumer,
            partition_id=0,
            polling_strategy=PollingStrategy.First(),
            count=3,
            auto_commit=True,
        )
        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=consumer, partition_id=0
        )

        assert _as_tuple(info) == (0, _last_offset(0), 2)

    @pytest.mark.asyncio
    async def test_store_consumer_offset_without_partition_id_is_rejected(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test the server needs an explicit partition to store an offset."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )

        with pytest.raises(RuntimeError, match="Invalid identifier"):
            await iggy_client.store_consumer_offset(
                stream_name,
                topic_name,
                consumer=Consumer.Single(unique_name()),
                offset=1,
            )

    @pytest.mark.asyncio
    async def test_store_consumer_offset_beyond_newest_message_is_rejected(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test an offset past the partition's newest message is refused."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        consumer = Consumer.Single(unique_name())

        with pytest.raises(RuntimeError, match="Invalid offset"):
            await iggy_client.store_consumer_offset(
                stream_name,
                topic_name,
                consumer=consumer,
                offset=_last_offset(0) + 1,
                partition_id=0,
            )
        assert (
            await iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=0
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_store_consumer_offset_on_an_empty_partition_is_rejected(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test even offset 0 is refused while the partition has no messages."""
        stream_name = unique_name()
        topic_name = unique_name()
        await iggy_client.create_stream(stream_name)
        await iggy_client.create_topic(
            stream=stream_name, name=topic_name, partitions_count=1
        )

        with pytest.raises(RuntimeError, match="Invalid offset"):
            await iggy_client.store_consumer_offset(
                stream_name,
                topic_name,
                consumer=Consumer.Single(unique_name()),
                offset=0,
                partition_id=0,
            )

    @pytest.mark.asyncio
    async def test_store_consumer_offset_passes_the_full_u64_range_to_the_server(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test the largest u64 offset converts and reaches the server.

        A conversion through a signed 64-bit integer would raise
        `OverflowError` before the request is sent.
        """
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )

        with pytest.raises(RuntimeError, match="Invalid offset"):
            await iggy_client.store_consumer_offset(
                stream_name,
                topic_name,
                consumer=Consumer.Single(unique_name()),
                offset=2**64 - 1,
                partition_id=0,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method", ["store", "get", "delete"], ids=["store", "get", "delete"]
    )
    async def test_consumer_offset_on_a_missing_partition_raises(
        self, iggy_client: IggyClient, unique_name, method
    ):
        """Test each offset call on a partition the topic lacks raises."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        consumer = Consumer.Single(unique_name())
        calls = {
            "store": lambda: iggy_client.store_consumer_offset(
                stream_name, topic_name, consumer=consumer, offset=0, partition_id=7
            ),
            "get": lambda: iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=7
            ),
            "delete": lambda: iggy_client.delete_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=7
            ),
        }

        with pytest.raises(RuntimeError, match="Partition .* was not found"):
            await calls[method]()

    @pytest.mark.asyncio
    async def test_consumer_offset_on_a_missing_stream_raises(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a missing stream is an error, not an absent offset."""
        with pytest.raises(RuntimeError, match="Stream .* was not found"):
            await iggy_client.get_consumer_offset(
                unique_name(),
                unique_name(),
                consumer=Consumer.Single(unique_name()),
                partition_id=0,
            )


class TestDeleteConsumerOffset:
    """Test deleting a stored offset."""

    @pytest.mark.asyncio
    async def test_delete_consumer_offset_removes_the_offset(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a deleted offset reads back `None` and `Next` starts over."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        consumer = Consumer.Single(unique_name())
        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=consumer, offset=2, partition_id=0
        )

        await iggy_client.delete_consumer_offset(
            stream_name, topic_name, consumer=consumer, partition_id=0
        )

        assert (
            await iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=0
            )
            is None
        )
        polled = await iggy_client.poll_messages(
            stream_name,
            topic_name,
            consumer=consumer,
            partition_id=0,
            polling_strategy=PollingStrategy.Next(),
            count=10,
            auto_commit=False,
        )
        assert [message.offset() for message in polled] == list(
            range(_last_offset(0) + 1)
        )

    @pytest.mark.asyncio
    async def test_delete_consumer_offset_leaves_other_partitions(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a delete on one partition keeps the offset on another."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name, partitions_count=2
        )
        consumer = Consumer.Single(unique_name())
        for partition_id in (0, 1):
            await iggy_client.store_consumer_offset(
                stream_name,
                topic_name,
                consumer=consumer,
                offset=partition_id + 1,
                partition_id=partition_id,
            )

        await iggy_client.delete_consumer_offset(
            stream_name, topic_name, consumer=consumer, partition_id=0
        )

        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=consumer, partition_id=1
        )
        assert _as_tuple(info) == (1, _last_offset(1), 2)

    @pytest.mark.asyncio
    async def test_delete_consumer_offset_without_a_stored_offset_raises(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test deleting an offset that does not exist is an error."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )

        with pytest.raises(RuntimeError, match="Consumer offset .* was not found"):
            await iggy_client.delete_consumer_offset(
                stream_name,
                topic_name,
                consumer=Consumer.Single(unique_name()),
                partition_id=0,
            )

    @pytest.mark.asyncio
    async def test_delete_consumer_offset_without_partition_id_is_rejected(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test the server needs an explicit partition to delete an offset."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        consumer = Consumer.Single(unique_name())
        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=consumer, offset=1, partition_id=0
        )

        with pytest.raises(RuntimeError, match="Invalid identifier"):
            await iggy_client.delete_consumer_offset(
                stream_name, topic_name, consumer=consumer
            )
        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=consumer, partition_id=0
        )
        assert _as_tuple(info) == (0, _last_offset(0), 1)


class TestConsumerGroupOffset:
    """Test offsets stored for a consumer group."""

    @pytest.mark.asyncio
    async def test_group_member_stores_gets_and_deletes_its_partition_offset(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a member round-trips a group offset kept apart from a consumer's."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        group_name = unique_name()
        await iggy_client.create_consumer_group(stream_name, topic_name, group_name)
        await iggy_client.join_consumer_group(stream_name, topic_name, group_name)
        group = Consumer.Group(group_name)

        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=group, offset=3, partition_id=0
        )

        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=group, partition_id=0
        )
        assert _as_tuple(info) == (0, _last_offset(0), 3)
        assert (
            await iggy_client.get_consumer_offset(
                stream_name,
                topic_name,
                consumer=Consumer.Single(group_name),
                partition_id=0,
            )
            is None
        )

        await iggy_client.delete_consumer_offset(
            stream_name, topic_name, consumer=group, partition_id=0
        )
        assert (
            await iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=group, partition_id=0
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_group_member_without_partition_id_reads_partition_zero_only(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test an omitted partition is not resolved to the member's assignment.

        Store and delete are rejected, and get reads partition 0, even though
        the sole member owns both partitions.
        """
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name, partitions_count=2
        )
        group_name = unique_name()
        await iggy_client.create_consumer_group(stream_name, topic_name, group_name)
        await iggy_client.join_consumer_group(stream_name, topic_name, group_name)
        group = Consumer.Group(group_name)
        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=group, offset=2, partition_id=1
        )

        with pytest.raises(RuntimeError, match="Invalid identifier"):
            await iggy_client.store_consumer_offset(
                stream_name, topic_name, consumer=group, offset=1
            )
        with pytest.raises(RuntimeError, match="Invalid identifier"):
            await iggy_client.delete_consumer_offset(
                stream_name, topic_name, consumer=group
            )
        assert (
            await iggy_client.get_consumer_offset(
                stream_name, topic_name, consumer=group
            )
            is None
        )

        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=group, offset=1, partition_id=0
        )
        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=group
        )
        assert _as_tuple(info) == (0, _last_offset(0), 1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["store", "delete"])
    async def test_group_offset_changes_require_membership(
        self, iggy_client: IggyClient, unique_name, method
    ):
        """Test a client outside the group cannot move or drop its offset."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        group_name = unique_name()
        await iggy_client.create_consumer_group(stream_name, topic_name, group_name)
        group = Consumer.Group(group_name)
        calls = {
            "store": lambda: iggy_client.store_consumer_offset(
                stream_name, topic_name, consumer=group, offset=1, partition_id=0
            ),
            "delete": lambda: iggy_client.delete_consumer_offset(
                stream_name, topic_name, consumer=group, partition_id=0
            ),
        }

        with pytest.raises(RuntimeError, match="does not own partition"):
            await calls[method]()

    @pytest.mark.asyncio
    async def test_group_offset_stays_readable_after_leaving_the_group(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test reads are not fenced, so a former member still sees the offset."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        group_name = unique_name()
        await iggy_client.create_consumer_group(stream_name, topic_name, group_name)
        await iggy_client.join_consumer_group(stream_name, topic_name, group_name)
        group = Consumer.Group(group_name)
        await iggy_client.store_consumer_offset(
            stream_name, topic_name, consumer=group, offset=2, partition_id=0
        )

        await iggy_client.leave_consumer_group(stream_name, topic_name, group_name)

        info = await iggy_client.get_consumer_offset(
            stream_name, topic_name, consumer=group, partition_id=0
        )
        assert _as_tuple(info) == (0, _last_offset(0), 2)
        with pytest.raises(RuntimeError, match="does not own partition"):
            await iggy_client.delete_consumer_offset(
                stream_name, topic_name, consumer=group, partition_id=0
            )

    @pytest.mark.asyncio
    async def test_get_consumer_offset_for_a_missing_group_returns_none(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a group that does not exist has no offset rather than an error."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )

        info = await iggy_client.get_consumer_offset(
            stream_name,
            topic_name,
            consumer=Consumer.Group(unique_name()),
            partition_id=0,
        )

        assert info is None


class TestConsumerOffsetArguments:
    """Test argument conversion, which fails before any request is sent."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("offset", [-1, 2**64], ids=["negative", "above-u64"])
    async def test_store_consumer_offset_rejects_offsets_outside_u64(
        self, iggy_client: IggyClient, unique_name, offset
    ):
        """Test an offset outside u64 fails conversion with OverflowError."""
        with pytest.raises(OverflowError):
            await iggy_client.store_consumer_offset(
                unique_name(),
                unique_name(),
                consumer=Consumer.Single(1),
                offset=offset,
                partition_id=0,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("offset", [1.0, "1", None], ids=["float", "str", "none"])
    async def test_store_consumer_offset_rejects_non_integer_offsets(
        self, iggy_client: IggyClient, unique_name, offset
    ):
        """Test an offset that is not an int fails conversion with TypeError."""
        with pytest.raises(TypeError):
            await iggy_client.store_consumer_offset(
                unique_name(),
                unique_name(),
                consumer=Consumer.Single(1),
                # pyrefly: ignore  # bad-argument-type
                offset=offset,
                partition_id=0,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["store", "get", "delete"])
    @pytest.mark.parametrize("partition_id", [-1, 2**32], ids=["negative", "above-u32"])
    async def test_consumer_offset_rejects_partition_ids_outside_u32(
        self, iggy_client: IggyClient, unique_name, method, partition_id
    ):
        """Test a partition id outside u32 fails conversion with OverflowError."""
        with pytest.raises(OverflowError):
            await _call_offset_method(
                iggy_client, unique_name, method, partition_id=partition_id
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["store", "get", "delete"])
    @pytest.mark.parametrize("consumer", ["single", 1], ids=["str", "int"])
    async def test_consumer_offset_rejects_a_non_consumer(
        self, iggy_client: IggyClient, unique_name, method, consumer
    ):
        """Test the consumer argument only accepts a Consumer."""
        with pytest.raises(TypeError, match="not an instance of 'Consumer'"):
            await _call_offset_method(
                iggy_client, unique_name, method, consumer=consumer
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["store", "get", "delete"])
    @pytest.mark.parametrize(
        "stream", [-1, 2**32, 1.5], ids=["negative", "above-u32", "float"]
    )
    async def test_consumer_offset_rejects_a_stream_that_is_not_an_identifier(
        self, iggy_client: IggyClient, unique_name, method, stream
    ):
        """Test a numeric stream id outside u32 raises TypeError, as elsewhere."""
        kwargs = {"consumer": Consumer.Single(1), "partition_id": 0}
        if method == "store":
            kwargs["offset"] = 0
        with pytest.raises(TypeError):
            await getattr(iggy_client, f"{method}_consumer_offset")(
                stream, unique_name(), **kwargs
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["store", "get", "delete"])
    async def test_consumer_offset_rejects_a_none_consumer(
        self, iggy_client: IggyClient, unique_name, method
    ):
        """Test `None` is not accepted in place of a Consumer."""
        kwargs = {"consumer": None, "partition_id": 0}
        if method == "store":
            kwargs["offset"] = 0
        with pytest.raises(TypeError, match="not an instance of 'Consumer'"):
            await getattr(iggy_client, f"{method}_consumer_offset")(
                unique_name(), unique_name(), **kwargs
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["store", "get", "delete"])
    @pytest.mark.parametrize("identifier", ["", "a" * 256], ids=["empty", "too-long"])
    async def test_consumer_offset_rejects_an_invalid_string_identifier(
        self, iggy_client: IggyClient, unique_name, method, identifier
    ):
        """Test a string consumer id is validated when it is converted."""
        with pytest.raises(ValueError, match="Invalid identifier"):
            await _call_offset_method(
                iggy_client, unique_name, method, consumer=Consumer.Single(identifier)
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["store", "get", "delete"])
    async def test_consumer_offset_takes_the_consumer_as_a_keyword(
        self, iggy_client: IggyClient, unique_name, method
    ):
        """Test the consumer cannot be passed positionally."""
        with pytest.raises(TypeError, match="positional argument"):
            await getattr(iggy_client, f"{method}_consumer_offset")(
                unique_name(), unique_name(), Consumer.Single(1)
            )


class TestConsumerOffsetOverHttp:
    """Test the offset calls over HTTP, which decode JSON instead of binary."""

    @pytest.mark.asyncio
    async def test_store_get_and_delete_consumer_offset(
        self, iggy_client: IggyClient, unique_name
    ):
        """Test a store, get and delete round trip through the HTTP API."""
        stream_name, topic_name = await _create_topic_with_messages(
            iggy_client, unique_name
        )
        host, port = get_http_server_config()
        http_client = IggyClient(HttpConfig(api_url=f"http://{host}:{port}"))
        await http_client.connect()
        await wait_for_ping(http_client)
        await http_client.login_user("iggy", "iggy")
        consumer = Consumer.Single(unique_name())

        try:
            assert (
                await http_client.get_consumer_offset(
                    stream_name, topic_name, consumer=consumer, partition_id=0
                )
                is None
            )
            await http_client.store_consumer_offset(
                stream_name, topic_name, consumer=consumer, offset=3, partition_id=0
            )
            info = await http_client.get_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=0
            )
            assert isinstance(info, ConsumerOffsetInfo)
            assert _as_tuple(info) == (0, _last_offset(0), 3)
            assert _as_tuple(
                await iggy_client.get_consumer_offset(
                    stream_name, topic_name, consumer=consumer, partition_id=0
                )
            ) == (0, _last_offset(0), 3)

            await http_client.delete_consumer_offset(
                stream_name, topic_name, consumer=consumer, partition_id=0
            )
            assert (
                await http_client.get_consumer_offset(
                    stream_name, topic_name, consumer=consumer, partition_id=0
                )
                is None
            )
        finally:
            await http_client.disconnect()
