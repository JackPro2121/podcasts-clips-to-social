import requests
from typing import List, Dict, Any, Optional
from src.config import BUFFER_ACCESS_TOKEN, BUFFER_CHANNEL_IDS

BUFFER_GRAPHQL_ENDPOINT = "https://api.buffer.com"

class BufferClient:
    def __init__(self, access_token: Optional[str] = None):
        self.token = access_token or BUFFER_ACCESS_TOKEN
        if not self.token:
            print("[-] Warning: BUFFER_ACCESS_TOKEN not configured.")

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def get_channels(self) -> List[Dict[str, Any]]:
        """Queries connected social channels from Buffer GraphQL using organization-based schema."""
        org_query = """
        query GetOrgs {
          account {
            organizations {
              id
              name
            }
          }
        }
        """
        try:
            res = requests.post(
                BUFFER_GRAPHQL_ENDPOINT,
                headers=self.headers,
                json={"query": org_query},
                timeout=15
            )
            if res.status_code != 200:
                print(f"[-] Buffer organizations query failed: {res.status_code} - {res.text}")
                return []

            orgs = res.json().get("data", {}).get("account", {}).get("organizations", [])
            channels = []
            for org in orgs:
                org_id = org.get("id")
                if not org_id:
                    continue
                ch_query = f"""query {{
                  channels(input: {{ organizationId: "{org_id}" }}) {{
                    id
                    name
                    service
                  }}
                }}"""
                ch_res = requests.post(
                    BUFFER_GRAPHQL_ENDPOINT,
                    headers=self.headers,
                    json={"query": ch_query},
                    timeout=15
                )
                if ch_res.status_code == 200:
                    ch_items = ch_res.json().get("data", {}).get("channels", [])
                    for ch in ch_items:
                        ch["organizationName"] = org.get("name")
                        channels.append(ch)
            return channels
        except Exception as e:
            print(f"[-] Error fetching Buffer channels: {e}")
            return []

    def schedule_video_post(
        self,
        video_url: str,
        text: str,
        channel_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Schedules video clip on specified or all connected Buffer channels.
        Uses the createPost mutation with assets [{ video: { url: ... } }].
        """
        if not self.token:
            print("[-] Cannot post to Buffer: BUFFER_ACCESS_TOKEN is missing.")
            return []

        target_channels = channel_ids or BUFFER_CHANNEL_IDS
        if not target_channels:
            print("[*] No channel IDs specified. Discovering connected channels from Buffer account...")
            connected = self.get_channels()
            if not connected:
                print("[-] No channels found in Buffer account. Connect your TikTok/Instagram/Shorts on Buffer.")
                return []
            target_channels = [ch["id"] for ch in connected]
            print(f"[+] Found {len(target_channels)} connected channel(s): {[ch['service'] for ch in connected]}")

        mutation = """
        mutation CreateVideoPost($input: CreatePostInput!) {
          createPost(input: $input) {
            ... on PostActionSuccess {
              post {
                id
                status
              }
            }
            ... on MutationError {
              message
            }
          }
        }
        """

        results = []
        for channel_id in target_channels:
            payload = {
                "query": mutation,
                "variables": {
                    "input": {
                        "channelId": channel_id,
                        "text": text,
                        "schedulingType": "automatic",
                        "mode": "addToQueue",
                        "assets": [
                            {
                                "video": {
                                    "url": video_url
                                }
                            }
                        ]
                    }
                }
            }

            try:
                print(f"[*] Dispatching video to Buffer channel: {channel_id}...")
                res = requests.post(
                    BUFFER_GRAPHQL_ENDPOINT,
                    headers=self.headers,
                    json=payload,
                    timeout=20
                )
                res_data = res.json()
                results.append({
                    "channel_id": channel_id,
                    "status_code": res.status_code,
                    "response": res_data
                })

                post_data = res_data.get("data", {}).get("createPost", {})
                if "post" in post_data:
                    print(f"[+] Successfully queued post on Buffer channel {channel_id}! Post ID: {post_data['post'].get('id')}")
                else:
                    error_msg = post_data.get("message", res.text)
                    print(f"[-] Buffer rejected post on {channel_id}: {error_msg}")

            except Exception as e:
                print(f"[-] Request error posting to Buffer channel {channel_id}: {e}")
                results.append({"channel_id": channel_id, "error": str(e)})

        return results
