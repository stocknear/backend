module.exports = function (fastify, opts, done) {
    const axios = opts.axios;

    fastify.get('/get-twitch-status', async (request, reply) => {
        let twitchStatus = false;

        const client_id = '5i041m3iztxuj0yx26scgzhri1etfi';
        const client_secret = '8p9gdmglz23lyc2nsrpbym5tpp15w0';
        const streamer_name = 'stocknear';

        try {
            // Obtain an access token from Twitch
            
            const tokenResponse = await axios.post('https://id.twitch.tv/oauth2/token', null, {
                params: {
                    client_id,
                    client_secret,
                    grant_type: 'client_credentials',
                },
            });

            const { access_token } = tokenResponse.data;

            // Check if the stream is online
            const streamResponse = await axios.get(
                `https://api.twitch.tv/helix/streams?user_login=${streamer_name}`,
                {
                    headers: {
                        'Client-ID': client_id,
                        Authorization: `Bearer ${access_token}`,
                    },
                }
            );

            const streamData = streamResponse.data;
            twitchStatus = streamData.data.length === 1;
            

        } catch (e) {
            console.error(e);
        }

        reply.send({ items: twitchStatus });
    });

    done();
};
