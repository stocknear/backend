import asyncio

import aiohttp

IMAGE_CHECK_CONCURRENCY = 15


def _looks_like_image(content_type):
    content_type = (content_type or '').lower()
    return not content_type or 'image' in content_type


class ImageValidator:
    """Async image reachability checker with simple caching."""

    def __init__(self, session, concurrency=IMAGE_CHECK_CONCURRENCY):
        self.session = session
        self.semaphore = asyncio.Semaphore(concurrency)
        self.cache = {}

    async def _head_request(self, url):
        try:
            async with self.semaphore:
                async with self.session.head(url, allow_redirects=True) as response:
                    if response.status == 200:
                        return _looks_like_image(response.headers.get('Content-Type'))
                    if response.status in (405, 501):
                        return None  # server rejects HEAD
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None
        return False

    async def _get_request(self, url):
        try:
            async with self.semaphore:
                headers = {'Range': 'bytes=0-0'}
                async with self.session.get(url, allow_redirects=True, headers=headers) as response:
                    if response.status in (200, 206):
                        return _looks_like_image(response.headers.get('Content-Type'))
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False
        return False

    async def _check_url(self, url):
        head_result = await self._head_request(url)
        if head_result is None:
            return await self._get_request(url)
        return head_result

    async def image_is_reachable(self, url):
        if not url:
            return False
        if url not in self.cache:
            self.cache[url] = await self._check_url(url)
        return self.cache[url]

    async def filter(self, data, image_key='image'):
        if not data:
            return []

        async def validate(item):
            image_url = item.get(image_key)
            if not image_url:
                return None
            return item if await self.image_is_reachable(image_url) else None

        validated_items = await asyncio.gather(*(validate(item) for item in data))
        return [item for item in validated_items if item]
